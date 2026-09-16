"""
solver/milp/branch_and_bound.py
--------------------------------
Branch-and-Cut MILP solver.

Integrates:
  - Root LP relaxation with cutting plane rounds (MIR, Cover, Clique, Gomory).
  - Primal heuristics: Rounding, Diving, Feasibility Pump, and RINS.
  - Branching: Reliability Pseudocost Branching with Strong Branching initialization.
  - Node relaxation warm-starting with dual simplex.
  - Hybrid node selection: best-bound priority queue with depth-first diving.
  - Root presolve reductions and postsolve reconstruction.

References:
    Wolsey (1998) "Integer Programming", §3.
    Achterberg (2009) "Constraint Integer Programming", PhD Thesis §3–10.
    Achterberg et al. (2005) "Branching rules revisited", Op. Res. Letters.
"""

from __future__ import annotations

import dataclasses
import heapq
import time
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    BB_TIME_LIMIT_S,
    CUT_VIOLATION_MIN,
    DEPTH_DIVE_FREQ,
    FEASIBILITY_TOL,
    HEURISTIC_INTERVAL,
    INTEGER_TOL,
    MAX_BB_NODES,
    MAX_CUTS_PER_ROUND,
    MAX_CUT_ROUNDS,
    OPTIMALITY_TOL,
    USE_PRESOLVE,
)
from solver.lp.simplex_dense import SolveResult
from solver.lp.simplex_revised import solve_lp_dual, solve_lp_revised
from solver.milp.branching import (
    PseudocostTable,
    branch_most_fractional,
    branch_reliability_pseudocost,
    branch_strong,
)
from solver.milp.cuts import add_cuts_to_problem, generate_all_cuts
from solver.milp.heuristics import (
    heuristic_diving,
    heuristic_feasibility_pump,
    heuristic_rins,
    heuristic_rounding,
)
from solver.presolve.postsolve import postsolve
from solver.presolve.presolve import presolve
from solver.problem import Problem


# ── Node data structure ───────────────────────────────────────────────────────

@dataclasses.dataclass
class BBNode:
    """One node in the B&B tree."""
    node_id:     int
    depth:       int
    lb_extra:    np.ndarray   # Extra lower bounds imposed on this node
    ub_extra:    np.ndarray   # Extra upper bounds imposed on this node
    lp_bound:    float = np.inf  # LP relaxation objective (minimisation)
    basis:       Optional[np.ndarray] = None  # Basis for warm start

    def __lt__(self, other):
        # For heap ordering: smaller bound = higher priority
        return self.lp_bound < other.lp_bound


# ── MILPSolveResult ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class MILPSolveResult:
    """Result of a MILP solve."""
    status:      str            # 'optimal' | 'infeasible' | 'node_limit' | 'time_limit'
    x:           Optional[np.ndarray]
    objective:   float
    nodes:       int            # Nodes processed
    lp_relaxation: float = np.inf  # Root LP relaxation bound
    gap:         float = 0.0    # Optimality gap at termination


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_node_problem(problem: Problem, lb_extra: np.ndarray, ub_extra: np.ndarray) -> Problem:
    """
    Create the LP relaxation for a B&B node by tightening variable bounds.
    All integer constraints are dropped (LP relaxation).
    """
    lb = np.maximum(problem.lb, lb_extra)
    ub = np.minimum(problem.ub, ub_extra)
    return Problem(
        c=problem.c,
        A_ub=problem.A_ub,
        b_ub=problem.b_ub,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        lb=lb,
        ub=ub,
        integer_mask=np.zeros(problem.n_vars, dtype=bool),  # LP relaxation
        sense=problem.sense,
        name=problem.name,
    )


def _is_integer_feasible(x: Optional[np.ndarray], mask: np.ndarray) -> bool:
    """Return True if all integer-constrained variables are integer-valued."""
    if x is None:
        return False
    if not mask.any():
        return True
    return bool(np.all(np.abs(x[mask] - np.round(x[mask])) <= INTEGER_TOL))


def _is_primal_feasible(problem: Problem, x: Optional[np.ndarray], tol: float = 1e-4) -> bool:
    """Return True if x satisfies all linear constraints and bounds of problem."""
    if x is None:
        return False
    if problem.n_ineq > 0:
        viol_ub = problem.A_ub @ x - problem.b_ub
        if np.any(viol_ub > tol):
            return False
    if problem.n_eq > 0:
        viol_eq = np.abs(problem.A_eq @ x - problem.b_eq)
        if np.any(viol_eq > tol):
            return False
    if np.any(x < problem.lb - tol) or np.any(x > problem.ub + tol):
        return False
    return True


def _best_bound(heap, fallback: float) -> float:
    """Peek at the best remaining lower bound from the heap."""
    return heap[0][0] if heap else fallback


def _compute_gap(incumbent: float, lb: float) -> float:
    """Relative optimality gap = |incumbent - lb| / (1 + |incumbent|)."""
    if incumbent == np.inf:
        return np.inf
    return abs(incumbent - lb) / (1.0 + abs(incumbent))


# ── Branch-and-Cut Solver ─────────────────────────────────────────────────────

def solve_milp(
    problem: Problem,
    time_limit: float = BB_TIME_LIMIT_S,
    node_limit: int = MAX_BB_NODES,
    verbose: bool = False,
    branch_strategy: str = "reliability",
    use_cuts: bool = True,
    use_heuristics: bool = True,
    use_presolve: bool = False,
) -> MILPSolveResult:
    """
    Solve a Mixed-Integer LP using Branch-and-Cut.

    Parameters
    ----------
    problem : Problem
        Must have integer_mask set for integer variables.
    time_limit : float
        Wall-clock time limit in seconds.
    node_limit : int
        Maximum B&B nodes to process.
    verbose : bool
        Print progress log.
    branch_strategy : str
        'reliability' | 'strong' | 'most_fractional'
    use_cuts : bool
        Generate cutting planes at root.
    use_heuristics : bool
        Run primal heuristics at root and periodic intervals.
    use_presolve : bool
        Run presolve reductions before search.

    Returns
    -------
    MILPSolveResult
    """
    t_start = time.monotonic()

    # ── Optional Presolve ──────────────────────────────────────────────────────
    if use_presolve and USE_PRESOLVE:
        pres_res = presolve(problem)
        if pres_res.infeasible:
            return MILPSolveResult("infeasible", None, np.inf, 0, np.inf, 0.0)

        # If all variables were eliminated
        if pres_res.problem.n_vars == 0:
            x_orig = postsolve(pres_res, np.zeros(0))
            obj = float(problem.c @ x_orig) if problem.sense == "min" else -float(problem.c @ x_orig)
            return MILPSolveResult("optimal", x_orig, obj, 0, obj, 0.0)

        if len(pres_res.actions) > 0:
            sub_res = solve_milp(
                pres_res.problem,
                time_limit=time_limit,
                node_limit=node_limit,
                verbose=verbose,
                branch_strategy=branch_strategy,
                use_cuts=use_cuts,
                use_heuristics=use_heuristics,
                use_presolve=False,
            )
            if sub_res.status == "optimal" and sub_res.x is not None:
                x_orig = postsolve(pres_res, sub_res.x)
                obj = float(problem.c @ x_orig) if problem.sense == "min" else -float(problem.c @ x_orig)
                return MILPSolveResult(
                    "optimal", x_orig, obj, sub_res.nodes, sub_res.lp_relaxation, sub_res.gap
                )
            elif sub_res.status == "infeasible":
                return MILPSolveResult("infeasible", None, np.inf, sub_res.nodes, np.inf, 0.0)
            else:
                x_orig = postsolve(pres_res, sub_res.x) if sub_res.x is not None else None
                obj = float(problem.c @ x_orig) if x_orig is not None else np.inf
                return MILPSolveResult(
                    sub_res.status, x_orig, obj, sub_res.nodes, sub_res.lp_relaxation, sub_res.gap
                )

    work_prob = problem
    n = work_prob.n_vars

    # ── Root LP relaxation ─────────────────────────────────────────────────────
    root_lp = _make_node_problem(work_prob, work_prob.lb.copy(), work_prob.ub.copy())
    root_result = solve_lp_revised(root_lp)

    if root_result.status == "infeasible":
        return MILPSolveResult("infeasible", None, np.inf, 1, np.inf, 0.0)
    if root_result.status == "unbounded":
        return MILPSolveResult("infeasible", None, -np.inf, 1, -np.inf, 0.0)

    root_bound = root_result.objective
    lp_relaxation = root_bound  # Store original LP bound

    # If no integer variables, LP relaxation IS the MILP solution
    if not work_prob.integer_mask.any():
        return MILPSolveResult(
            "optimal", root_result.x, root_result.objective, 1, lp_relaxation, 0.0
        )

    # Check if root LP is already integer-feasible
    if _is_integer_feasible(root_result.x, work_prob.integer_mask):
        return MILPSolveResult(
            "optimal", root_result.x.copy(), root_result.objective, 1, lp_relaxation, 0.0
        )

    # Incumbent (best integer-feasible solution found so far)
    incumbent_x: Optional[np.ndarray] = None
    incumbent_obj: float = np.inf

    # ── Root Cut Generation Rounds ─────────────────────────────────────────────
    if use_cuts:
        for cut_round in range(min(MAX_CUT_ROUNDS, 5)):
            if time.monotonic() - t_start >= time_limit:
                break
            cuts = generate_all_cuts(
                work_prob,
                root_result.x,
                basis=root_result.basis,
                max_total=MAX_CUTS_PER_ROUND,
            )
            if not cuts:
                break

            work_prob = add_cuts_to_problem(work_prob, cuts)
            new_root_lp = _make_node_problem(work_prob, work_prob.lb.copy(), work_prob.ub.copy())

            new_root = solve_lp_dual(new_root_lp, basis=root_result.basis)
            if new_root.status != "optimal":
                new_root = solve_lp_revised(new_root_lp)

            if new_root.status == "infeasible":
                return MILPSolveResult("infeasible", None, np.inf, 1, np.inf, 0.0)

            if new_root.status == "optimal":
                improvement = new_root.objective - root_result.objective
                root_result = new_root
                if _is_integer_feasible(root_result.x, work_prob.integer_mask) and _is_primal_feasible(work_prob, root_result.x):
                    incumbent_x = root_result.x.copy()
                    incumbent_obj = root_result.objective
                    return MILPSolveResult(
                        "optimal", incumbent_x, incumbent_obj, 1, lp_relaxation, 0.0
                    )
                if improvement < 1e-4:
                    break

    # ── Root Primal Heuristics ─────────────────────────────────────────────────
    if use_heuristics and (time.monotonic() - t_start < time_limit):
        # 1. Simple Rounding
        x_round = heuristic_rounding(work_prob)
        if x_round is not None and _is_primal_feasible(work_prob, x_round):
            obj_round = float(work_prob.c @ x_round) if work_prob.sense == "min" else -float(work_prob.c @ x_round)
            if obj_round < incumbent_obj - OPTIMALITY_TOL:
                incumbent_obj = obj_round
                incumbent_x = x_round.copy()

        # 2. Diving Heuristic
        x_dive = heuristic_diving(work_prob, max_dives=25)
        if x_dive is not None and _is_primal_feasible(work_prob, x_dive):
            obj_dive = float(work_prob.c @ x_dive) if work_prob.sense == "min" else -float(work_prob.c @ x_dive)
            if obj_dive < incumbent_obj - OPTIMALITY_TOL:
                incumbent_obj = obj_dive
                incumbent_x = x_dive.copy()

        # 3. Feasibility Pump if no incumbent found yet
        if incumbent_x is None:
            x_fp = heuristic_feasibility_pump(work_prob)
            if x_fp is not None and _is_primal_feasible(work_prob, x_fp):
                obj_fp = float(work_prob.c @ x_fp) if work_prob.sense == "min" else -float(work_prob.c @ x_fp)
                if obj_fp < incumbent_obj - OPTIMALITY_TOL:
                    incumbent_obj = obj_fp
                    incumbent_x = x_fp.copy()

    # ── Priority queue: (lp_bound, node_id, node) ─────────────────────────────
    node_counter = [0]

    def _new_node(depth, lb_extra, ub_extra, lp_bound, basis=None) -> BBNode:
        node_counter[0] += 1
        return BBNode(node_counter[0], depth, lb_extra, ub_extra, lp_bound, basis)

    root_node = _new_node(0, work_prob.lb.copy(), work_prob.ub.copy(), root_result.objective, root_result.basis)
    heap: List = []
    heapq.heappush(heap, (root_result.objective, root_node.node_id, root_node))

    pseudocosts = PseudocostTable(work_prob.n_vars)
    nodes_processed = 0
    dive_child: Optional[BBNode] = None

    while (heap or dive_child is not None) and nodes_processed < node_limit:
        # Check time limit
        if time.monotonic() - t_start > time_limit:
            gap = _compute_gap(incumbent_obj, _best_bound(heap, lp_relaxation))
            return MILPSolveResult(
                "time_limit", incumbent_x, incumbent_obj,
                nodes_processed, lp_relaxation, gap
            )

        # Hybrid node selection: dive child or best-bound from heap
        if dive_child is not None:
            node = dive_child
            dive_child = None
        else:
            _, _, node = heapq.heappop(heap)

        nodes_processed += 1

        # Prune by bound
        if node.lp_bound >= incumbent_obj - OPTIMALITY_TOL:
            continue

        # ── Solve LP relaxation at this node ──────────────────────────────────
        if node.node_id == 1:
            lp_res = root_result
        else:
            node_prob = _make_node_problem(work_prob, node.lb_extra, node.ub_extra)
            # Try warm start with dual simplex
            if node.basis is not None:
                lp_res = solve_lp_dual(node_prob, basis=node.basis)
                if lp_res.status not in ("optimal", "infeasible"):
                    lp_res = solve_lp_revised(node_prob)
            else:
                lp_res = solve_lp_revised(node_prob)

        if lp_res.status in ("infeasible", "unbounded"):
            continue  # Prune infeasible subtree

        lp_obj = lp_res.objective

        # Prune by bound
        if lp_obj >= incumbent_obj - OPTIMALITY_TOL:
            continue

        x = lp_res.x

        # ── Integer feasibility check ──────────────────────────────────────────
        if _is_integer_feasible(x, work_prob.integer_mask) and _is_primal_feasible(work_prob, x):
            if lp_obj < incumbent_obj - OPTIMALITY_TOL:
                incumbent_obj = lp_obj
                incumbent_x = x.copy()
                if verbose:
                    print(f"  [incumbent] node {node.node_id} obj={incumbent_obj:.6f}")
            continue

        # ── Periodic Heuristics ────────────────────────────────────────────────
        if use_heuristics and nodes_processed % HEURISTIC_INTERVAL == 0:
            if incumbent_x is not None:
                x_rins = heuristic_rins(work_prob, x, incumbent_x)
                if x_rins is not None and _is_primal_feasible(work_prob, x_rins):
                    obj_rins = float(work_prob.c @ x_rins) if work_prob.sense == "min" else -float(work_prob.c @ x_rins)
                    if obj_rins < incumbent_obj - OPTIMALITY_TOL:
                        incumbent_obj = obj_rins
                        incumbent_x = x_rins.copy()

        # ── Branching Variable Selection ───────────────────────────────────────
        if branch_strategy == "reliability":
            branch_var, _, _ = branch_reliability_pseudocost(
                x, work_prob.integer_mask, work_prob,
                node.lb_extra, node.ub_extra, lp_obj, pseudocosts
            )
            if branch_var < 0:
                branch_var = branch_most_fractional(x, work_prob.integer_mask)
        elif branch_strategy == "strong":
            branch_var, _, _ = branch_strong(
                x, work_prob.integer_mask, work_prob,
                node.lb_extra, node.ub_extra, lp_obj, pseudocosts
            )
            if branch_var < 0:
                branch_var = branch_most_fractional(x, work_prob.integer_mask)
        else:
            branch_var = branch_most_fractional(x, work_prob.integer_mask)

        if branch_var < 0:
            continue  # No fractional variables found

        x_val = x[branch_var]
        floor_val = np.floor(x_val)
        ceil_val  = np.ceil(x_val)

        # Left child: branch_var ≤ floor(x_val)
        lb_left = node.lb_extra.copy()
        ub_left = node.ub_extra.copy()
        ub_left[branch_var] = min(ub_left[branch_var], floor_val)

        # Right child: branch_var ≥ ceil(x_val)
        lb_right = node.lb_extra.copy()
        ub_right = node.ub_extra.copy()
        lb_right[branch_var] = max(lb_right[branch_var], ceil_val)

        children = []
        for lb_c, ub_c in [(lb_left, ub_left), (lb_right, ub_right)]:
            if np.any(lb_c > ub_c + FEASIBILITY_TOL):
                continue
            child = _new_node(node.depth + 1, lb_c, ub_c, lp_obj, basis=lp_res.basis)
            children.append(child)

        # Depth-first diving logic:
        # Every DEPTH_DIVE_FREQ nodes, dive into the child whose variable was closer to integer
        if children:
            if (nodes_processed % DEPTH_DIVE_FREQ != 0) and len(children) == 2:
                # Normal best-first: push both
                for c in children:
                    heapq.heappush(heap, (c.lp_bound, c.node_id, c))
            elif len(children) == 1:
                heapq.heappush(heap, (children[0].lp_bound, children[0].node_id, children[0]))
            else:
                # Dive into one, push other to heap
                frac = x_val - floor_val
                if frac < 0.5:
                    dive_child = children[0]
                    heapq.heappush(heap, (children[1].lp_bound, children[1].node_id, children[1]))
                else:
                    dive_child = children[1]
                    heapq.heappush(heap, (children[0].lp_bound, children[0].node_id, children[0]))

    # ── Termination ───────────────────────────────────────────────────────────
    if incumbent_x is None:
        return MILPSolveResult("infeasible", None, np.inf, nodes_processed, lp_relaxation, 0.0)

    best_remaining = _best_bound(heap, lp_relaxation)
    gap = _compute_gap(incumbent_obj, best_remaining)
    status = "optimal" if not heap or gap < OPTIMALITY_TOL else "node_limit"

    # Only return original problem variable slice if cuts were appended
    final_x = incumbent_x[:problem.n_vars] if len(incumbent_x) > problem.n_vars else incumbent_x
    final_obj = float(problem.c @ final_x) if problem.sense == "min" else -float(problem.c @ final_x)

    return MILPSolveResult(status, final_x, final_obj, nodes_processed, lp_relaxation, gap)
