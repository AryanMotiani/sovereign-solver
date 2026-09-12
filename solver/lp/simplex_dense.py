"""
solver/lp/simplex_dense.py
---------------------------
Dense full-tableau simplex method — correctness scaffold only.

This is intentionally simple (dense NumPy matrices, no sparse tricks,
no performance concern).  Its sole purpose is to validate LP logic on
small toy problems before we invest in the revised sparse simplex.

Algorithm:
  - Two-phase Simplex (Phase I uses Big-M to find BFS, Phase II optimises)
  - Dantzig pivot rule (most negative reduced cost)
  - Bland's rule after BLAND_RULE_THRESHOLD iterations (anti-cycling)
  - Standard minimum-ratio test with Bland's tie-breaking

Returns SolveResult namedtuple.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from solver.config import (
    FEASIBILITY_TOL,
    OPTIMALITY_TOL,
    PIVOT_TOL,
    BLAND_RULE_THRESHOLD,
    MAX_SIMPLEX_ITERS,
)
from solver.problem import Problem


# ── Result ────────────────────────────────────────────────────────────────────

class SolveResult:
    """Holds the output of a solver call."""

    def __init__(
        self,
        status: str,
        x: Optional[np.ndarray],
        objective: float,
        iterations: int,
        message: str = "",
        basis: Optional[np.ndarray] = None,
        gaps: Optional[list] = None,
    ):
        self.status = status        # 'optimal' | 'infeasible' | 'unbounded' | 'iteration_limit'
        self.x = x                  # solution vector (None if not available)
        self.objective = objective  # objective value at x (original sense)
        self.iterations = iterations
        self.message = message
        self.basis = basis          # final basis (ndarray of col indices) for warm-starting
        self.gaps = gaps or []      # duality gaps per IPM iteration

    def __repr__(self) -> str:
        obj_str = f"{self.objective:.6g}" if self.objective is not None else "N/A"
        return (
            f"SolveResult(status={self.status!r}, obj={obj_str}, "
            f"iters={self.iterations})"
        )


# ── Dense Simplex ─────────────────────────────────────────────────────────────

def solve_lp_dense(problem: Problem, big_m: float = 1e6) -> SolveResult:
    """
    Solve an LP using the dense full-tableau simplex (Big-M method).

    The problem is converted to standard form:
        min  cᵀx
        s.t. Ax = b,  x ≥ 0

    Steps:
      1. Convert inequalities to equalities by adding slack variables.
      2. Handle variable bounds (shift variables: x' = x - lb).
      3. Add Big-M artificial variables for equality rows without an obvious
         basic feasible solution (BFS).
      4. Run simplex; detect optimality, infeasibility, unboundedness.

    Parameters
    ----------
    problem : Problem
        Must be an LP (is_milp and is_qp both False, but we don't strictly
        enforce this here — callers should ensure it).
    big_m : float
        Penalty coefficient for artificial variables in Big-M method.

    Returns
    -------
    SolveResult
    """
    # ── Standard-form conversion ──────────────────────────────────────────────
    # Map original variables x → x' = x - lb  (shift so x' ≥ 0)
    lb = problem.lb.copy()
    ub = problem.ub.copy()
    c_orig = problem.c.copy()
    n_orig = problem.n_vars

    # For variables with lb = -inf, use lb = 0 (free variables split into x+ - x-)
    # Simplified: treat -inf lb as 0 for the dense scaffold (tiny toy problems only)
    lb_finite = np.where(np.isfinite(lb), lb, 0.0)
    ub_shifted = ub - lb_finite  # shifted upper bounds

    # Adjust RHS for the shift
    b_ub_shifted = problem.b_ub - problem.A_ub.dot(lb_finite)
    b_eq_shifted = problem.b_eq - problem.A_eq.dot(lb_finite)

    # Adjust objective for the shift
    c_shifted = c_orig.copy()
    c_obj_offset = float(c_orig @ lb_finite)  # constant term in objective

    # ── Build equality system: Ax = b, x ≥ 0 ─────────────────────────────────
    # Rows: [inequalities (with slacks), equalities]
    # Variables: [original (shifted), slack (ineq), artificial (eq + any ineq with -rhs)]

    m_ub = problem.n_ineq
    m_eq = problem.n_eq
    m_total = m_ub + m_eq
    n_slacks = m_ub
    n_artificials = m_eq  # for equality rows

    # Count how many inequality rows have negative RHS after shift
    # (those need an artificial too, after flipping the row)
    neg_rhs_ineq = []
    for i in range(m_ub):
        if b_ub_shifted[i] < -FEASIBILITY_TOL:
            neg_rhs_ineq.append(i)
    n_art_ineq = len(neg_rhs_ineq)
    n_art_total = n_artificials + n_art_ineq

    n_total = n_orig + n_slacks + n_art_total  # total variables in standard form

    # Build dense tableau: rows = constraints, cols = variables + 1 (RHS)
    T = np.zeros((m_total + 1, n_total + 1))  # last row = objective

    # Fill original variable columns
    if m_ub > 0:
        T[:m_ub, :n_orig] = problem.A_ub.toarray()
    if m_eq > 0:
        T[m_ub:m_ub + m_eq, :n_orig] = problem.A_eq.toarray()

    # Fill slack columns (one per inequality)
    for i in range(m_ub):
        rhs_i = b_ub_shifted[i]
        if rhs_i < -FEASIBILITY_TOL:
            # Flip the row so RHS ≥ 0
            T[i, :] = -T[i, :]
            T[i, n_orig + i] = -1.0  # slack becomes -1 after flip
        else:
            T[i, n_orig + i] = 1.0

    # Fill RHS
    T[:m_ub, -1] = np.abs(b_ub_shifted)   # after row flipping, all ≥ 0
    T[m_ub:m_ub + m_eq, -1] = b_eq_shifted

    # Fill artificial variable columns
    art_start = n_orig + n_slacks
    art_idx = 0
    for i in neg_rhs_ineq:  # artificials for flipped ineq rows
        T[i, art_start + art_idx] = 1.0
        art_idx += 1
    for j in range(m_eq):  # artificials for equality rows
        T[m_ub + j, art_start + art_idx] = 1.0
        art_idx += 1

    # Initial basis: slacks for non-flipped ineq rows + artificials elsewhere
    basis = list(range(n_orig, n_orig + n_slacks))  # slack vars initially
    for i in neg_rhs_ineq:
        basis[i] = art_start + neg_rhs_ineq.index(i)
    for j in range(m_eq):
        basis.append(art_start + n_art_ineq + j)

    # Objective row: Big-M on artificials, original costs on originals
    T[-1, :n_orig] = c_shifted
    T[-1, art_start:art_start + n_art_total] = big_m

    # Make objective row relative to initial basis (eliminate basic artificial costs)
    for i, b in enumerate(basis):
        if b >= art_start:  # artificial in basis → eliminate from obj row
            T[-1, :] -= big_m * T[i, :]

    # ── Simplex iterations ────────────────────────────────────────────────────
    iters = 0

    while iters < MAX_SIMPLEX_ITERS:
        obj_row = T[-1, :-1]

        # Pivot column selection (Dantzig or Bland)
        if iters > BLAND_RULE_THRESHOLD:
            # Bland's rule: smallest index with negative reduced cost
            neg_mask = obj_row < -OPTIMALITY_TOL
            if not neg_mask.any():
                break
            pivot_col = int(np.where(neg_mask)[0][0])
        else:
            # Dantzig: most negative reduced cost
            pivot_col = int(np.argmin(obj_row))
            if obj_row[pivot_col] >= -OPTIMALITY_TOL:
                break  # optimal

        # Minimum ratio test
        col_vals = T[:-1, pivot_col]
        rhs_vals = T[:-1, -1]

        positive_mask = col_vals > PIVOT_TOL
        if not positive_mask.any():
            return SolveResult("unbounded", None, -np.inf, iters, "Unbounded problem detected.")

        ratios = np.where(positive_mask, rhs_vals / col_vals, np.inf)

        if iters > BLAND_RULE_THRESHOLD:
            # Bland's tie-breaking: smallest basis index
            min_ratio = ratios.min()
            candidates = np.where(np.abs(ratios - min_ratio) < 1e-12)[0]
            pivot_row = int(candidates[np.argmin([basis[r] for r in candidates])])
        else:
            pivot_row = int(np.argmin(ratios))

        # Pivot
        pivot_val = T[pivot_row, pivot_col]
        T[pivot_row, :] /= pivot_val
        for i in range(T.shape[0]):
            if i != pivot_row:
                T[i, :] -= T[i, pivot_col] * T[pivot_row, :]

        basis[pivot_row] = pivot_col
        iters += 1

    else:
        return SolveResult(
            "iteration_limit", None, np.inf, iters,
            f"Max iterations ({MAX_SIMPLEX_ITERS}) reached."
        )

    # ── Check for infeasibility (artificials still in basis) ──────────────────
    for i, b in enumerate(basis):
        if b >= art_start and abs(T[i, -1]) > FEASIBILITY_TOL:
            return SolveResult("infeasible", None, np.inf, iters, "Problem is infeasible.")

    # ── Extract solution ──────────────────────────────────────────────────────
    x_std = np.zeros(n_total)
    for i, b in enumerate(basis):
        if b < n_total:
            x_std[b] = T[i, -1]

    x_shifted = x_std[:n_orig]
    x_orig = x_shifted + lb_finite

    # Clip to bounds (numerical noise)
    x_orig = np.clip(x_orig, lb_finite, np.where(np.isfinite(ub), ub, x_orig))

    obj_val = float(c_orig @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult("optimal", x_orig, obj_val, iters)
