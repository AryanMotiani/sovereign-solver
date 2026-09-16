"""
solver/milp/branch_and_bound.py
--------------------------------
Branch-and-Cut MILP solver — full pipeline.

Algorithm:
  1. Presolve (optional): reduce problem size before B&B.
  2. Root LP relaxation.
  3. Primal heuristics at root: rounding + feasibility pump to get early incumbent.
  4. Cut loop at root node: MIR + cover + clique cuts injected into LP.
  5. Best-first B&B with advanced branching (pseudocost / strong branching).
  6. At each node: solve LP, check integer feasibility, branch.
  7. Postsolve: reconstruct original-space solution after presolve.

Branching variable selection:
  - Uses PseudocostTable + reliability branching (Achterberg 2005).
  - Falls back to strong branching for unreliable variables.
  - Falls back to most-fractional when pseudocosts unavailable.

Node selection: best-first (priority queue on LP bound).

Pruning: prune if LP relaxation >= current incumbent (minimisation).

References:
    Wolsey (1998) "Integer Programming", §3.
    Achterberg (2009) PhD Thesis, §3–4, §7–8.
"""

from __future__ import annotations

import heapq
import time
import dataclasses
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from solver.config import (
    BB_TIME_LIMIT_S,
    FEASIBILITY_TOL,
    INTEGER_TOL,
    MAX_BB_NODES,
    MAX_CUT_ROUNDS,
    OPTIMALITY_TOL,
)
from solver.lp.simplex_dense import SolveResult
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem


# ── Node data structure ───────────────────────────────────────────────────────

@dataclasses.dataclass
class BBNode:
    """One node in the B&B tree.  Immutable after creation."""
    node_id:     int
    depth:       int
    lb_extra:    np.ndarray   # Extra lower bounds imposed on this node
    ub_extra:    np.ndarray   # Extra upper bounds imposed on this node
    lp_bound:    float = np.inf  # LP relaxation objective (minimisation)

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


def _is_integer_feasible(x: np.ndarray, mask: np.ndarray) -> bool:
    """Return True if all integer-constrained variables are integer-valued."""
    if not mask.any():
        return True
    return np.all(np.abs(x[mask] - np.round(x[mask])) <= INTEGER_TOL)


def _most_fractional_var(x: np.ndarray, mask: np.ndarray) -> int:
    """
    Select the branching variable: integer-masked variable with fractional
    part closest to 0.5 (most fractional / maximum infeasibility rule).
    Returns index into x, or -1 if none is fractional.
    """
    int_vars = np.where(mask)[0]
    fracs = np.abs(x[int_vars] - np.round(x[int_vars]))
    frac_mask = fracs > INTEGER_TOL
    if not frac_mask.any():
        return -1
    candidates = int_vars[frac_mask]
    frac_vals = fracs[frac_mask]
    best = candidates[np.argmax(np.minimum(frac_vals, 1.0 - frac_vals))]
    return int(best)


def _apply_root_cuts(problem: Problem, x_star: np.ndarray, max_rounds: int = MAX_CUT_ROUNDS) -> Problem:
    """
    Apply cut generation rounds at the root node.
    Returns an augmented problem with cuts appended to A_ub.
    """
    try:
        from solver.milp.cuts import generate_all_cuts, add_cuts_to_problem
    except ImportError:
        return problem

    current_prob = problem
    current_x = x_star

    for _ in range(max_rounds):
        cuts = generate_all_cuts(current_prob, current_x)
        if not cuts:
            break
        current_prob = add_cuts_to_problem(current_prob, cuts)
        # Re-solve to get a new LP solution with cuts applied
        lp_res = solve_lp_revised(_make_node_problem(
            current_prob, current_prob.lb.copy(), current_prob.ub.copy()
        ))
        if lp_res.status != "optimal":
            break
        current_x = lp_res.x

    return current_prob


def _try_primal_heuristics(problem: Problem, x_lp: np.ndarray,
                            incumbent_x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Try rounding, diving, feasibility pump, and RINS (if incumbent available).
    Return the best new incumbent found, or None.
    """
    try:
        from solver.milp.heuristics import (
            heuristic_rounding,
            heuristic_diving,
            heuristic_feasibility_pump,
            heuristic_rins,
        )
        from solver.utils.feasibility import is_feasible
    except ImportError:
        return None

    best_x = None
    best_obj = np.inf if incumbent_x is None else float(problem.c @ incumbent_x)

    def _try(x_cand):
        nonlocal best_x, best_obj
        if x_cand is not None and is_feasible(problem, x_cand):
            obj = float(problem.c @ x_cand)
            if obj < best_obj - OPTIMALITY_TOL:
                best_x = x_cand.copy()
                best_obj = obj

    _try(heuristic_rounding(problem))
    _try(heuristic_diving(problem))
    _try(heuristic_feasibility_pump(problem))
    if incumbent_x is not None:
        _try(heuristic_rins(problem, x_lp, incumbent_x))

    return best_x


def _select_branch_var(problem: Problem, x: np.ndarray,
                       lb_extra: np.ndarray, ub_extra: np.ndarray,
                       parent_obj: float, pseudocosts) -> int:
    """
    Choose branching variable using pseudocost / strong branching.
    Falls back to most-fractional if no pseudocost table supplied.
    """
    if pseudocosts is None:
        return _most_fractional_var(x, problem.integer_mask)

    try:
        from solver.milp.branching import branch_reliability_pseudocost
        branch_var, _, _ = branch_reliability_pseudocost(
            x, problem.integer_mask, problem, lb_extra, ub_extra, parent_obj, pseudocosts
        )
        return branch_var if branch_var >= 0 else _most_fractional_var(x, problem.integer_mask)
    except Exception:
        return _most_fractional_var(x, problem.integer_mask)


# ── Branch-and-Cut ────────────────────────────────────────────────────────────

def solve_milp(
    problem: Problem,
    time_limit: float = BB_TIME_LIMIT_S,
    node_limit: int = MAX_BB_NODES,
    use_presolve: bool = True,
    use_cuts: bool = True,
    use_heuristics: bool = True,
    use_advanced_branching: bool = True,
    verbose: bool = False,
) -> MILPSolveResult:
    """
    Solve a Mixed-Integer LP using Branch-and-Cut.

    Pipeline:
      1. Presolve (optional) — reduces problem size.
      2. Root LP relaxation.
      3. Root-level primal heuristics (optional) — fast incumbent.
      4. Root-level cut rounds (optional) — tighten LP relaxation.
      5. B&B tree with pseudocost branching (optional).
      6. Postsolve (if presolve was applied) — reconstruct full solution.

    Parameters
    ----------
    problem : Problem
        Must have integer_mask set for integer variables.
    time_limit : float
        Wall-clock time limit in seconds.
    node_limit : int
        Maximum B&B nodes to process.
    use_presolve : bool
        Apply presolve reductions before B&B.
    use_cuts : bool
        Apply cut rounds at root node.
    use_heuristics : bool
        Run primal heuristics at root for an early incumbent.
    use_advanced_branching : bool
        Use pseudocost / reliability branching instead of most-fractional.
    verbose : bool
        Print progress log.

    Returns
    -------
    MILPSolveResult
    """
    t_start = time.monotonic()

    # ── Step 1: Presolve ──────────────────────────────────────────────────────
    presolve_result = None
    col_map = None
    working_problem = problem

    if use_presolve and problem.is_milp:
        try:
            from solver.presolve.presolve import presolve
            from solver.presolve.postsolve import postsolve as _postsolve
            presolve_result = presolve(working_problem)
            if presolve_result.infeasible:
                return MILPSolveResult("infeasible", None, np.inf, 0, np.inf, 0.0)
            working_problem = presolve_result.problem
            col_map = presolve_result.col_map
            if verbose and (presolve_result.n_fixed > 0 or presolve_result.n_rows_removed > 0):
                print(f"  [presolve] fixed={presolve_result.n_fixed} "
                      f"rows_removed={presolve_result.n_rows_removed} "
                      f"bounds_tightened={presolve_result.n_bound_tightened}")
        except Exception:
            presolve_result = None
            working_problem = problem

    n = working_problem.n_vars

    # ── Step 2: Root LP relaxation ────────────────────────────────────────────
    root_lp = _make_node_problem(working_problem, working_problem.lb.copy(), working_problem.ub.copy())
    root_result = solve_lp_revised(root_lp)

    if root_result.status == "infeasible":
        return MILPSolveResult("infeasible", None, np.inf, 1, np.inf, 0.0)
    if root_result.status == "unbounded":
        return MILPSolveResult("infeasible", None, -np.inf, 1, -np.inf, 0.0)

    root_bound = root_result.objective

    # Compute LP relaxation bound in ORIGINAL problem space for reporting.
    # If presolve was applied, we re-solve the root LP on the original problem
    # to get a bound that's comparable to the MILP objective in original space.
    if presolve_result is not None and presolve_result.n_fixed + presolve_result.n_rows_removed > 0:
        orig_root_lp = _make_node_problem(problem, problem.lb.copy(), problem.ub.copy())
        orig_root_res = solve_lp_revised(orig_root_lp)
        lp_relaxation = orig_root_res.objective if orig_root_res.status == "optimal" else root_bound
    else:
        lp_relaxation = root_bound

    # If no integer variables, LP relaxation IS the MILP solution
    if not working_problem.integer_mask.any():
        x_reduced = root_result.x
        x_full = _postsolve_solution(presolve_result, x_reduced, problem)
        obj = float(problem.c @ x_full)
        return MILPSolveResult("optimal", x_full, obj, 1, lp_relaxation, 0.0)

    # Incumbent tracking — two parallel trackers:
    #   incumbent_obj/x   : in ORIGINAL problem space (for reporting)
    #   _prune_obj        : in WORKING (presolved) problem space (for pruning)
    incumbent_x: Optional[np.ndarray] = None
    incumbent_obj: float = np.inf
    _prune_obj: float = np.inf   # presolved-space bound for B&B pruning

    def _update_incumbent(x_reduced: np.ndarray, lp_obj_presolved: float = np.inf) -> bool:
        """Map x_reduced to original space, recompute obj, update if better."""
        nonlocal incumbent_x, incumbent_obj, _prune_obj
        x_full_cand = _postsolve_solution(presolve_result, x_reduced, problem)
        if x_full_cand is None:
            return False
        obj_cand = float(problem.c @ x_full_cand)
        if obj_cand < incumbent_obj - OPTIMALITY_TOL:
            incumbent_obj = obj_cand
            incumbent_x = x_full_cand
            # Update presolved-space pruning bound
            # Use lp_obj_presolved when available, else fall back to original obj
            _prune_obj = lp_obj_presolved if lp_obj_presolved < np.inf else obj_cand
            return True
        return False

    # Check if root LP is already integer-feasible
    if _is_integer_feasible(root_result.x, working_problem.integer_mask):
        _update_incumbent(root_result.x, root_result.objective)
        return MILPSolveResult("optimal", incumbent_x, incumbent_obj, 1, lp_relaxation, 0.0)

    # ── Step 3: Primal heuristics at root ─────────────────────────────────────
    if use_heuristics:
        h_x = _try_primal_heuristics(working_problem, root_result.x, None)
        if h_x is not None:
            if _update_incumbent(h_x) and verbose:
                print(f"  [heuristic] root incumbent obj={incumbent_obj:.6f}")

    # ── Step 4: Root-level cut rounds ─────────────────────────────────────────
    if use_cuts and working_problem.is_milp:
        working_problem_cut = _apply_root_cuts(working_problem, root_result.x)
        if working_problem_cut is not working_problem:
            # Re-solve root LP with cuts to get tighter internal bound
            root_lp_cut = _make_node_problem(
                working_problem_cut,
                working_problem_cut.lb.copy(),
                working_problem_cut.ub.copy()
            )
            root_result_cut = solve_lp_revised(root_lp_cut)
            if root_result_cut.status == "optimal":
                working_problem = working_problem_cut
                root_result = root_result_cut
                root_bound = root_result.objective
                if verbose:
                    print(f"  [cuts] root bound tightened: {lp_relaxation:.6f} -> {root_bound:.6f}")
                # Note: lp_relaxation (original-space root LP) is unchanged;
                # root_bound is used internally for B&B pruning

    # ── Step 5: B&B tree ──────────────────────────────────────────────────────
    # Set up pseudocost table for advanced branching
    pseudocosts = None
    if use_advanced_branching:
        try:
            from solver.milp.branching import PseudocostTable
            pseudocosts = PseudocostTable(n)
        except Exception:
            pseudocosts = None

    node_counter = [0]
    def _new_node(depth, lb_extra, ub_extra, lp_bound) -> BBNode:
        node_counter[0] += 1
        return BBNode(node_counter[0], depth, lb_extra, ub_extra, lp_bound)

    root_node = _new_node(0, working_problem.lb.copy(), working_problem.ub.copy(), root_bound)
    heap: List = []
    heapq.heappush(heap, (root_bound, root_node.node_id, root_node))

    nodes_processed = 0

    while heap and nodes_processed < node_limit:
        # Check time limit
        if time.monotonic() - t_start > time_limit:
            gap = _compute_gap(incumbent_obj, _best_bound(heap, lp_relaxation))
            return MILPSolveResult(
                "time_limit", incumbent_x, incumbent_obj,
                nodes_processed, lp_relaxation, gap
            )

        _, _, node = heapq.heappop(heap)
        nodes_processed += 1

        # Prune by bound (minimisation) — compare in presolved problem space
        if node.lp_bound >= _prune_obj - OPTIMALITY_TOL:
            if verbose:
                print(f"  [prune] node {node.node_id} lb={node.lp_bound:.4f} >= prune={_prune_obj:.4f}")
            continue

        # ── Solve LP relaxation at this node ──────────────────────────────────
        node_prob = _make_node_problem(working_problem, node.lb_extra, node.ub_extra)
        lp_res = solve_lp_revised(node_prob)

        if lp_res.status in ("infeasible", "unbounded"):
            continue  # Prune: infeasible subtree

        lp_obj = lp_res.objective

        # Prune by bound (presolved space)
        if lp_obj >= _prune_obj - OPTIMALITY_TOL:
            continue

        x = lp_res.x

        # ── Integer feasibility check ──────────────────────────────────────────
        if _is_integer_feasible(x, working_problem.integer_mask):
            if _update_incumbent(x, lp_obj) and verbose:
                print(f"  [incumbent] node {node.node_id} obj={incumbent_obj:.6f}")
            continue

        # ── Branch ────────────────────────────────────────────────────────────
        branch_var = _select_branch_var(
            working_problem, x, node.lb_extra, node.ub_extra, lp_obj, pseudocosts
        )
        if branch_var < 0:
            continue  # No fractional variables (shouldn't happen)

        x_val = x[branch_var]
        floor_val = np.floor(x_val)
        ceil_val  = np.ceil(x_val)

        # Left child: branch_var <= floor(x_val)
        lb_left = node.lb_extra.copy()
        ub_left = node.ub_extra.copy()
        ub_left[branch_var] = min(ub_left[branch_var], floor_val)

        # Right child: branch_var >= ceil(x_val)
        lb_right = node.lb_extra.copy()
        ub_right = node.ub_extra.copy()
        lb_right[branch_var] = max(lb_right[branch_var], ceil_val)

        for lb_c, ub_c in [(lb_left, ub_left), (lb_right, ub_right)]:
            if np.any(lb_c > ub_c + FEASIBILITY_TOL):
                continue  # Infeasible child: skip
            child = _new_node(node.depth + 1, lb_c, ub_c, lp_obj)
            heapq.heappush(heap, (lp_obj, child.node_id, child))

    # ── Termination ───────────────────────────────────────────────────────────
    if incumbent_x is None:
        return MILPSolveResult("infeasible", None, np.inf, nodes_processed, lp_relaxation, 0.0)

    best_remaining = _best_bound(heap, lp_relaxation)
    gap = _compute_gap(incumbent_obj, best_remaining)
    status = "optimal" if not heap or gap < OPTIMALITY_TOL else "node_limit"

    # incumbent_x and incumbent_obj are already in original problem space
    return MILPSolveResult(status, incumbent_x, incumbent_obj, nodes_processed, lp_relaxation, gap)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _postsolve_solution(
    presolve_result,
    x_reduced: Optional[np.ndarray],
    original_problem: Problem,
) -> Optional[np.ndarray]:
    """
    Apply postsolve to map a reduced-space solution back to original space.
    If presolve was not applied, returns x_reduced unchanged.
    """
    if x_reduced is None:
        return None
    if presolve_result is None:
        return x_reduced

    try:
        from solver.presolve.postsolve import postsolve
        return postsolve(presolve_result, x_reduced)
    except Exception:
        return x_reduced


def _best_bound(heap, fallback: float) -> float:
    """Peek at the best remaining lower bound from the heap."""
    return heap[0][0] if heap else fallback


def _compute_gap(incumbent: float, lb: float) -> float:
    """Relative optimality gap = (incumbent - lb) / (1 + |incumbent|)."""
    if incumbent == np.inf:
        return np.inf
    return abs(incumbent - lb) / (1.0 + abs(incumbent))
