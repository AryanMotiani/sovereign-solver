"""
solver/milp/branch_and_bound.py
--------------------------------
Branch-and-Bound MILP solver.

Algorithm:
  - LP relaxation is solved at each node using `solve_lp_revised`.
  - Branching variable: most-fractional heuristic (closest to 0.5).
  - Node selection: best-first (priority queue on LP bound).
  - Pruning: prune if LP relaxation ≥ current incumbent (minimisation).
  - Termination: all nodes processed or time/node limit reached.

Data Structures:
  - `BBNode`: immutable frozen dataclass carrying the LP problem for this node
    plus bound info.
  - `heapq` min-heap on (lower_bound, node_id) for best-first traversal.

References:
    Wolsey (1998) "Integer Programming", §3.
    Achterberg (2009) PhD Thesis, §3–4.
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
    # Closest to 0.5
    best = candidates[np.argmax(np.minimum(frac_vals, 1.0 - frac_vals))]
    return int(best)


# ── Branch-and-Bound ──────────────────────────────────────────────────────────

def solve_milp(
    problem: Problem,
    time_limit: float = BB_TIME_LIMIT_S,
    node_limit: int = MAX_BB_NODES,
    verbose: bool = False,
) -> MILPSolveResult:
    """
    Solve a Mixed-Integer LP using Branch-and-Bound.

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

    Returns
    -------
    MILPSolveResult
    """
    t_start = time.monotonic()
    n = problem.n_vars

    # ── Root LP relaxation ─────────────────────────────────────────────────────
    root_lp = _make_node_problem(problem, problem.lb.copy(), problem.ub.copy())
    root_result = solve_lp_revised(root_lp)

    if root_result.status == "infeasible":
        return MILPSolveResult("infeasible", None, np.inf, 1, np.inf, 0.0)
    if root_result.status == "unbounded":
        return MILPSolveResult("infeasible", None, -np.inf, 1, -np.inf, 0.0)

    root_bound = root_result.objective
    lp_relaxation = root_bound

    # If no integer variables, LP relaxation IS the MILP solution
    if not problem.integer_mask.any():
        return MILPSolveResult(
            "optimal", root_result.x, root_result.objective, 1, lp_relaxation, 0.0
        )

    # Incumbent (best integer-feasible solution found so far)
    incumbent_x: Optional[np.ndarray] = None
    incumbent_obj: float = np.inf

    # Check if root LP is already integer-feasible
    if _is_integer_feasible(root_result.x, problem.integer_mask):
        incumbent_x = root_result.x.copy()
        incumbent_obj = root_result.objective
        gap = 0.0
        return MILPSolveResult("optimal", incumbent_x, incumbent_obj, 1, lp_relaxation, gap)

    # ── Priority queue: (lp_bound, node_id, node) ─────────────────────────────
    node_counter = [0]
    def _new_node(depth, lb_extra, ub_extra, lp_bound) -> BBNode:
        node_counter[0] += 1
        return BBNode(node_counter[0], depth, lb_extra, ub_extra, lp_bound)

    root_node = _new_node(0, problem.lb.copy(), problem.ub.copy(), root_bound)
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

        # Prune by bound (minimisation)
        if node.lp_bound >= incumbent_obj - OPTIMALITY_TOL:
            if verbose:
                print(f"  [prune] node {node.node_id} lb={node.lp_bound:.4f} >= inc={incumbent_obj:.4f}")
            continue

        # ── Solve LP relaxation at this node ──────────────────────────────────
        node_prob = _make_node_problem(problem, node.lb_extra, node.ub_extra)
        lp_res = solve_lp_revised(node_prob)

        if lp_res.status in ("infeasible", "unbounded"):
            continue  # Prune: infeasible subtree

        lp_obj = lp_res.objective

        # Prune by bound
        if lp_obj >= incumbent_obj - OPTIMALITY_TOL:
            continue

        x = lp_res.x

        # ── Integer feasibility check ──────────────────────────────────────────
        if _is_integer_feasible(x, problem.integer_mask):
            if lp_obj < incumbent_obj - OPTIMALITY_TOL:
                incumbent_obj = lp_obj
                incumbent_x = x.copy()
                if verbose:
                    print(f"  [incumbent] node {node.node_id} obj={incumbent_obj:.6f}")
            continue

        # ── Branch ────────────────────────────────────────────────────────────
        branch_var = _most_fractional_var(x, problem.integer_mask)
        if branch_var < 0:
            continue  # No fractional variables (shouldn't happen)

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

    return MILPSolveResult(status, incumbent_x, incumbent_obj, nodes_processed, lp_relaxation, gap)


def _best_bound(heap, fallback: float) -> float:
    """Peek at the best remaining lower bound from the heap."""
    return heap[0][0] if heap else fallback


def _compute_gap(incumbent: float, lb: float) -> float:
    """Relative optimality gap = (incumbent - lb) / (1 + |incumbent|)."""
    if incumbent == np.inf:
        return np.inf
    return abs(incumbent - lb) / (1.0 + abs(incumbent))
