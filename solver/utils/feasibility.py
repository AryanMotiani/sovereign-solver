"""
solver/utils/feasibility.py
----------------------------
Shared feasibility checker used by ALL solver modules before accepting
any solution as an incumbent or returning it to the user.

The invariant:  is_feasible(problem, x) must be called on EVERY
solution returned by a heuristic.  Never bypass this check.
"""

from __future__ import annotations

import numpy as np

from solver.problem import Problem
from solver.config import FEASIBILITY_TOL, INTEGER_TOL


def is_feasible(
    problem: Problem,
    x: np.ndarray,
    tol: float = FEASIBILITY_TOL,
    int_tol: float = INTEGER_TOL,
    verbose: bool = False,
) -> bool:
    """
    Return True iff x satisfies ALL constraints of `problem`.

    Checks (in order — all must pass):
      1. Variable bounds:        lb - tol ≤ x ≤ ub + tol
      2. Inequality constraints: A_ub @ x ≤ b_ub + tol  (element-wise)
      3. Equality constraints:   |A_eq @ x - b_eq| ≤ tol (element-wise)
      4. Integrality:            |x[i] - round(x[i])| ≤ int_tol  for integer vars

    Parameters
    ----------
    problem : Problem
        The optimization problem instance.
    x : ndarray, shape (n_vars,)
        Candidate solution vector.
    tol : float
        Constraint violation tolerance (default from config.FEASIBILITY_TOL).
    int_tol : float
        Integer feasibility tolerance (default from config.INTEGER_TOL).
    verbose : bool
        If True, print the first violated constraint before returning False.

    Returns
    -------
    bool
        True if x is feasible, False otherwise.
    """
    x = np.asarray(x, dtype=float)

    if x.shape != (problem.n_vars,):
        if verbose:
            print(
                f"[is_feasible] Shape mismatch: x.shape={x.shape}, "
                f"expected ({problem.n_vars},)"
            )
        return False

    # ── 1. Variable bounds ────────────────────────────────────────────────────
    lb_viol = np.where(x < problem.lb - tol)[0]
    if lb_viol.size > 0:
        if verbose:
            i = lb_viol[0]
            print(
                f"[is_feasible] Lower-bound violation: x[{i}]={x[i]:.6g} "
                f"< lb[{i}]={problem.lb[i]:.6g}"
            )
        return False

    ub_viol = np.where(x > problem.ub + tol)[0]
    if ub_viol.size > 0:
        if verbose:
            i = ub_viol[0]
            print(
                f"[is_feasible] Upper-bound violation: x[{i}]={x[i]:.6g} "
                f"> ub[{i}]={problem.ub[i]:.6g}"
            )
        return False

    # ── 2. Inequality constraints  A_ub x ≤ b_ub ─────────────────────────────
    if problem.n_ineq > 0:
        ineq_lhs = problem.A_ub.dot(x)
        ineq_viol = np.where(ineq_lhs > problem.b_ub + tol)[0]
        if ineq_viol.size > 0:
            if verbose:
                i = ineq_viol[0]
                print(
                    f"[is_feasible] Inequality violation: row {i}: "
                    f"A_ub[{i}]@x={ineq_lhs[i]:.6g} > b_ub[{i}]={problem.b_ub[i]:.6g}"
                )
            return False

    # ── 3. Equality constraints  A_eq x = b_eq ───────────────────────────────
    if problem.n_eq > 0:
        eq_res = np.abs(problem.A_eq.dot(x) - problem.b_eq)
        eq_viol = np.where(eq_res > tol)[0]
        if eq_viol.size > 0:
            if verbose:
                i = eq_viol[0]
                print(
                    f"[is_feasible] Equality violation: row {i}: "
                    f"|A_eq[{i}]@x - b_eq[{i}]|={eq_res[i]:.6g}"
                )
            return False

    # ── 4. Integrality ────────────────────────────────────────────────────────
    if problem.is_milp:
        int_vars = np.where(problem.integer_mask)[0]
        int_res = np.abs(x[int_vars] - np.round(x[int_vars]))
        int_viol = np.where(int_res > int_tol)[0]
        if int_viol.size > 0:
            if verbose:
                idx = int_vars[int_viol[0]]
                print(
                    f"[is_feasible] Integrality violation: x[{idx}]={x[idx]:.6g} "
                    f"(fractional part {int_res[int_viol[0]]:.6g})"
                )
            return False

    return True


def violation_summary(problem: Problem, x: np.ndarray, tol: float = FEASIBILITY_TOL) -> dict:
    """
    Return a dict summarising all constraint violations for diagnostics.
    Useful in test failure messages.
    """
    x = np.asarray(x, dtype=float)
    summary: dict = {}

    lb_viol = np.maximum(0.0, problem.lb - x - tol)
    ub_viol = np.maximum(0.0, x - problem.ub - tol)
    summary["max_lb_violation"] = float(lb_viol.max()) if lb_viol.size else 0.0
    summary["max_ub_violation"] = float(ub_viol.max()) if ub_viol.size else 0.0

    if problem.n_ineq > 0:
        ineq_viol = np.maximum(0.0, problem.A_ub.dot(x) - problem.b_ub - tol)
        summary["max_ineq_violation"] = float(ineq_viol.max())
    else:
        summary["max_ineq_violation"] = 0.0

    if problem.n_eq > 0:
        eq_viol = np.abs(problem.A_eq.dot(x) - problem.b_eq)
        summary["max_eq_violation"] = float(eq_viol.max())
    else:
        summary["max_eq_violation"] = 0.0

    if problem.is_milp:
        int_vars = np.where(problem.integer_mask)[0]
        int_res = np.abs(x[int_vars] - np.round(x[int_vars]))
        summary["max_int_violation"] = float(int_res.max()) if int_res.size else 0.0
    else:
        summary["max_int_violation"] = 0.0

    summary["feasible"] = all(v == 0.0 for v in summary.values())
    return summary
