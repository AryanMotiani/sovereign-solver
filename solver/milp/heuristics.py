"""
solver/milp/heuristics.py
--------------------------
Primal heuristics for MILP: quick integer-feasible solutions.

Implements four heuristics:
  1. Simple Rounding    — round fractional LP solution to nearest integer
  2. Diving             — fix variables one by one, re-solve LP
  3. Feasibility Pump   — project between integer and LP-feasible spaces
  4. RINS               — fix variables that agree between relaxation & incumbent,
                          solve restricted sub-MIP

All heuristics return an Optional[np.ndarray] — the feasible solution found,
or None if none was found within the budget.

References:
    Bertacco et al. (2007) "A feasibility pump heuristic for general MIPs"
    Achterberg (2009) PhD Thesis §10.
    Danna et al. (2005) "Exploring relaxation induced neighborhoods to
      improve MIP solutions" — RINS.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from solver.config import (
    FEASIBILITY_TOL,
    INTEGER_TOL,
    MAX_FP_ITERS,
    RINS_NODE_LIMIT,
)
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem
from solver.utils.feasibility import is_feasible


# ── Helpers ───────────────────────────────────────────────────────────────────

def _round_solution(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Round integer variables to nearest integer."""
    x_out = x.copy()
    x_out[mask] = np.round(x_out[mask])
    return x_out


def _lp_solve(problem: Problem) -> Optional[np.ndarray]:
    """Solve LP relaxation; return solution vector or None."""
    res = solve_lp_revised(problem)
    return res.x if res.status == "optimal" else None


def _check_feasible(x: np.ndarray, problem: Problem) -> bool:
    """Full feasibility check including integrality."""
    if x is None:
        return False
    if not is_feasible(problem, x):  # problem is first arg
        return False
    mask = problem.integer_mask
    if mask.any():
        return np.all(np.abs(x[mask] - np.round(x[mask])) <= INTEGER_TOL)
    return True


# ── 1. Simple rounding ────────────────────────────────────────────────────────

def heuristic_rounding(problem: Problem) -> Optional[np.ndarray]:
    """
    Solve LP relaxation and round integer variables to nearest integer.
    Returns the rounded solution if feasible, else None.
    """
    lp_prob = Problem(
        c=problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub,
        A_eq=problem.A_eq, b_eq=problem.b_eq,
        lb=problem.lb, ub=problem.ub,
        integer_mask=np.zeros(problem.n_vars, dtype=bool),
        sense=problem.sense, name=problem.name,
    )
    x_lp = _lp_solve(lp_prob)
    if x_lp is None:
        return None

    x_rounded = _round_solution(x_lp, problem.integer_mask)
    # Clamp to bounds
    x_rounded = np.clip(x_rounded, problem.lb, problem.ub)
    return x_rounded if _check_feasible(x_rounded, problem) else None


# ── 2. Diving heuristic ───────────────────────────────────────────────────────

def heuristic_diving(problem: Problem, max_dives: int = 50) -> Optional[np.ndarray]:
    """
    Fractional diving: iteratively fix the most-fractional integer variable
    to its nearest integer bound and re-solve the LP.

    Stops when:
     - Integer-feasible solution found → return it.
     - LP becomes infeasible → backtrack once, then give up.
     - max_dives rounds exhausted.
    """
    lb = problem.lb.copy()
    ub = problem.ub.copy()

    for _ in range(max_dives):
        node_prob = Problem(
            c=problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub,
            A_eq=problem.A_eq, b_eq=problem.b_eq,
            lb=lb, ub=ub,
            integer_mask=np.zeros(problem.n_vars, dtype=bool),
            sense=problem.sense, name=problem.name,
        )
        x_lp = _lp_solve(node_prob)
        if x_lp is None:
            return None  # Infeasible node

        mask = problem.integer_mask
        fracs = np.abs(x_lp[mask] - np.round(x_lp[mask]))
        if np.all(fracs <= INTEGER_TOL):
            # All integer variables are integer-valued
            x_rounded = _round_solution(x_lp, mask)
            return x_rounded if _check_feasible(x_rounded, problem) else None

        # Fix the most-fractional variable
        int_idx = np.where(mask)[0]
        best_local = int(np.argmax(np.minimum(fracs, 1.0 - fracs)))
        j = int_idx[best_local]
        x_j = x_lp[j]

        # Round down (choose the floor direction)
        ub[j] = min(ub[j], np.floor(x_j))

    return None


# ── 3. Feasibility Pump ───────────────────────────────────────────────────────

def heuristic_feasibility_pump(problem: Problem) -> Optional[np.ndarray]:
    """
    Feasibility Pump (Bertacco et al. 2007):

    Alternates between:
      T-step: round x to nearest integer → x̄
      P-step: solve LP minimizing ‖x - x̄‖₁

    Terminates when x = x̄ (found integer-feasible point) or MAX_FP_ITERS reached.
    """
    n = problem.n_vars
    mask = problem.integer_mask

    # Start with LP relaxation solution
    lp_x = _lp_solve(Problem(
        c=problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub,
        A_eq=problem.A_eq, b_eq=problem.b_eq,
        lb=problem.lb, ub=problem.ub,
        integer_mask=np.zeros(n, dtype=bool),
        sense=problem.sense, name=problem.name,
    ))
    if lp_x is None:
        return None

    x = lp_x.copy()

    for _ in range(MAX_FP_ITERS):
        # T-step: round integer variables
        x_bar = _round_solution(x, mask)

        # Check if x_bar is feasible
        if _check_feasible(x_bar, problem):
            return x_bar

        # P-step: solve  min ‖x - x̄‖₁  s.t. Ax = b, lb ≤ x ≤ ub
        # Reformulate: min Σ d_j  where d_j ≥ x_j - x̄_j, d_j ≥ x̄_j - x_j
        # Only penalise integer variables to stay close to x̄.
        # This is equivalent to min cᵀ_fp x  where c_fp[j] = sign(x[j] - x_bar[j])

        c_fp = np.zeros(n)
        for j in np.where(mask)[0]:
            diff = x[j] - x_bar[j]
            c_fp[j] = 1.0 if diff >= 0 else -1.0

        pump_prob = Problem(
            c=c_fp,
            A_ub=problem.A_ub, b_ub=problem.b_ub,
            A_eq=problem.A_eq, b_eq=problem.b_eq,
            lb=problem.lb, ub=problem.ub,
            integer_mask=np.zeros(n, dtype=bool),
            sense="min",
            name=problem.name + "_pump",
        )
        x_new = _lp_solve(pump_prob)
        if x_new is None:
            return None

        x = x_new

    return None


# ── 4. RINS ───────────────────────────────────────────────────────────────────

def heuristic_rins(
    problem: Problem,
    x_lp: np.ndarray,
    x_inc: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Relaxation Induced Neighborhood Search (Danna et al. 2005).

    Fix all integer variables where x_lp[j] ≈ x_inc[j] (within INTEGER_TOL).
    Solve the restricted sub-MIP on the remaining free variables.

    Parameters
    ----------
    problem : Problem
    x_lp    : Current LP relaxation solution (fractional)
    x_inc   : Current incumbent (integer-feasible)

    Returns
    -------
    Better integer solution or None.
    """
    if x_inc is None:
        return None

    mask = problem.integer_mask
    lb = problem.lb.copy()
    ub = problem.ub.copy()
    new_int_mask = mask.copy()

    n_fixed = 0
    for j in np.where(mask)[0]:
        # Fix variables that agree between LP relaxation and incumbent
        if abs(x_lp[j] - x_inc[j]) <= INTEGER_TOL:
            lb[j] = x_inc[j]
            ub[j] = x_inc[j]
            new_int_mask[j] = False
            n_fixed += 1

    if n_fixed == 0:
        return None  # Nothing to fix

    # Solve restricted problem (sub-MILP with fewer integer vars)
    from solver.milp.branch_and_bound import solve_milp
    sub_prob = Problem(
        c=problem.c,
        A_ub=problem.A_ub, b_ub=problem.b_ub,
        A_eq=problem.A_eq, b_eq=problem.b_eq,
        lb=lb, ub=ub,
        integer_mask=new_int_mask,
        sense=problem.sense,
        name=problem.name + "_rins",
    )

    result = solve_milp(sub_prob, node_limit=RINS_NODE_LIMIT)
    if result.status == "optimal" and result.x is not None:
        if result.objective < float(problem.c @ x_inc) - FEASIBILITY_TOL:
            return result.x

    return None
