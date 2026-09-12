"""
solver/lp/simplex_revised.py
-----------------------------
Production-quality revised simplex LP solver — primal and dual variants.

Algorithm summary
-----------------
PRIMAL REVISED SIMPLEX
  Maintains a basis B (index set of basic variables) and its LU factorization.
  Each iteration:
    1. Solve   B y = cᴮ              → y  (dual variables / simplex multipliers)
    2. Compute reduced costs: rc_j = c_j - yᵀ a_j  for non-basic j
    3. Pivot column: min reduced cost (Dantzig) or min index (Bland)
    4. Solve   B d = a_s              → d  (simplex direction)
    5. Minimum ratio test → leaving variable r
    6. Pivot: update basis, re-factorize every REFACTORIZE_EVERY steps

DUAL REVISED SIMPLEX
  Starts dual-feasible (all rc ≥ 0 for minimisation), primal-infeasible.
  Each iteration:
    1. Find leaving variable: most infeasible basic variable
    2. Solve   Bᵀ u = eᵣ             → u  (dual simplex direction in constraint space)
    3. Dual ratio test: choose entering variable s = argmin |rc_j / (uᵀ a_j)|
    4. Pivot: update basis
  Used for warm-starting after a bound change (B&B branching).

Standard form
-------------
Input problem is converted to:
    min  cᵀx
    s.t. [A_ub | I_slack] [x; s] = b_rhs,  x ≥ 0, s ≥ 0
where equalities are handled by splitting: each equality row gets a free
artificial for Phase I and the equality is maintained through the basis.

References
----------
- Vanderbei, "Linear Programming: Foundations and Extensions" (4th ed.)
- Huangfu & Hall, Math. Prog. Computation 10(2), 2018 (dual simplex design)
- Suhl & Suhl, Annals of OR 43 (1993) (fast LU update)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    BLAND_RULE_THRESHOLD,
    FEASIBILITY_TOL,
    MAX_SIMPLEX_ITERS,
    OPTIMALITY_TOL,
    PIVOT_TOL,
    REFACTORIZE_EVERY,
)
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem
from solver.utils.sparse_lu import SparseLU


def _build_standard_form(problem: Problem):
    """
    Convert LP to standard equality form for revised simplex.

    Strategy
    --------
    For each inequality row Aᵢx ≤ bᵢ:
      - bᵢ ≥ 0: add slack sᵢ ≥ 0 → Aᵢx + sᵢ = bᵢ.  Initial BFS: sᵢ = bᵢ.
      - bᵢ < 0: multiply by -1 to get -Aᵢx ≥ -bᵢ (i.e. -bᵢ > 0).
                Add surplus sᵢ ≥ 0 and artificial aᵢ:  -Aᵢx - sᵢ + aᵢ = -bᵢ.
    For each equality row:
      Add artificial only.

    Returns
    -------
    c_aug     : objective (n_aug,) — Big-M on artificials
    A_aug     : csr_matrix (m_total, n_aug)
    b_aug     : rhs (m_total,)  — all ≥ 0
    basis0    : initial basis of length m_total (slacks + artificials)
    lb_finite : shifted lower bounds of original variables
    n_orig    : number of original (shifted) variables
    n_aug     : total number of columns in augmented system
    art_start : column index where artificials begin
    n_art     : number of artificial variables
    BIG_M     : Big-M penalty value used
    """
    BIG_M = 1e6
    n_orig = problem.n_vars
    m_ub = problem.n_ineq
    m_eq = problem.n_eq

    # Shift variables: x' = x - lb ≥ 0
    lb = problem.lb.copy()
    ub = problem.ub.copy()
    lb_finite = np.where(np.isfinite(lb), lb, 0.0)
    b_ub_s = problem.b_ub - problem.A_ub.dot(lb_finite)
    b_eq_s = problem.b_eq - problem.A_eq.dot(lb_finite)
    c_s = problem.c.copy()

    # Add upper bound rows: for finite ub[j], add x'_j ≤ ub[j] - lb[j]
    ub_shift = ub - lb_finite
    finite_ub_mask = np.isfinite(ub) & np.isfinite(lb_finite)
    # Only add bounds that are actually tighter than the default (infinite)
    finite_ub_cols = np.where(finite_ub_mask)[0]
    if len(finite_ub_cols) > 0:
        A_ub_extra = np.zeros((len(finite_ub_cols), n_orig))
        for k, j in enumerate(finite_ub_cols):
            A_ub_extra[k, j] = 1.0
        b_ub_extra = ub_shift[finite_ub_cols]
        # Stack onto existing A_ub / b_ub
        A_ub_orig = problem.A_ub.toarray() if m_ub > 0 else np.zeros((0, n_orig))
        A_ub_combined = np.vstack([A_ub_orig, A_ub_extra])
        b_ub_combined = np.concatenate([b_ub_s, b_ub_extra])
        # Build problem attrs for this function
        _A_ub = A_ub_combined
        _b_ub = b_ub_combined
        m_ub = m_ub + len(finite_ub_cols)
    else:
        _A_ub = problem.A_ub.toarray() if problem.n_ineq > 0 else np.zeros((0, n_orig))
        _b_ub = b_ub_s

    m_total = m_ub + m_eq

    # Determine which ineq rows need an artificial
    neg_ub_rows = np.where(_b_ub < -FEASIBILITY_TOL)[0]
    pos_ub_rows = np.setdiff1d(np.arange(m_ub), neg_ub_rows)

    n_slacks = m_ub          # one slack per ineq row
    n_art_ub = len(neg_ub_rows)
    n_art_eq = m_eq
    n_art = n_art_ub + n_art_eq
    art_start = n_orig + n_slacks
    n_aug = n_orig + n_slacks + n_art

    # Build row by row
    rows_data = []  # list of (row_vec, rhs_val)
    basis0 = np.zeros(m_total, dtype=int)

    # Use the combined A_ub (with upper-bound rows added) and b_ub
    A_orig_ub = _A_ub  # already a numpy dense array (shape m_ub × n_orig)
    A_orig_eq = problem.A_eq.toarray() if m_eq > 0 else np.zeros((0, n_orig))

    art_col = art_start  # running column index for next artificial

    for i in range(m_ub):
        row = np.zeros(n_aug)
        if _b_ub[i] >= -FEASIBILITY_TOL:
            # Normal case: slack forms initial basis
            row[:n_orig] = A_orig_ub[i]
            row[n_orig + i] = 1.0   # slack
            rows_data.append((row, _b_ub[i]))
            basis0[i] = n_orig + i  # slack is basic
        else:
            # Negative RHS: flip row, add surplus (-slack) + artificial
            row[:n_orig] = -A_orig_ub[i]
            row[n_orig + i] = -1.0  # surplus
            row[art_col] = 1.0      # artificial
            rows_data.append((row, -_b_ub[i]))
            basis0[i] = art_col
            art_col += 1

    for j in range(m_eq):
        row = np.zeros(n_aug)
        row[:n_orig] = A_orig_eq[j]
        row[art_col] = 1.0          # artificial
        rows_data.append((row, b_eq_s[j]))
        basis0[m_ub + j] = art_col
        art_col += 1

    if rows_data:
        A_aug = np.array([r for r, _ in rows_data])
        b_aug = np.array([v for _, v in rows_data])
    else:
        A_aug = np.zeros((0, n_aug))
        b_aug = np.zeros(0)

    # Objective: original + Big-M on artificials
    c_aug = np.concatenate([c_s, np.zeros(n_slacks), np.full(n_art, BIG_M)])

    # Eliminate artificials from objective row (so initial reduced costs are correct)
    for i, bi in enumerate(basis0):
        if bi >= art_start:
            c_aug -= BIG_M * A_aug[i]  # A_aug is a dense numpy array at this point

    return (
        c_aug,
        sp.csr_matrix(A_aug),
        b_aug,
        basis0,
        lb_finite,
        n_orig,
        n_aug,
        art_start,
        n_art,
        BIG_M,
    )



# ── Revised Primal Simplex ────────────────────────────────────────────────────

def solve_lp_revised(
    problem: Problem,
    basis: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    use_bland: bool = False,
) -> SolveResult:
    """
    Solve an LP with the revised primal simplex (Big-M method).

    Parameters
    ----------
    problem : Problem
    basis : optional external starting basis (ignored unless provided by dual)
    use_bland : force Bland's rule throughout
    """
    (c_aug, A_aug, b_aug, basis0, lb_finite,
     n_orig, n_aug, art_start, n_art, BIG_M) = _build_standard_form(problem)

    m = A_aug.shape[0]

    # Edge case: no constraints → optimal at lower bound (x' = 0)
    if m == 0 or n_aug == 0:
        x_orig = lb_finite.copy()
        obj_val = float(problem.c @ x_orig)
        if problem.sense == "max":
            obj_val = -obj_val
        return SolveResult("optimal", x_orig, obj_val, 0)

    if basis is None:
        basis = basis0.copy()
    else:
        basis = basis.copy().astype(int)

    # ── Factorize initial basis ───────────────────────────────────────────────
    def _bmat():
        return A_aug[:, basis]

    try:
        lu = SparseLU(_bmat())
    except ValueError as e:
        return SolveResult("infeasible", None, np.inf, 0, str(e))

    iters = 0
    pivot_count_since_refact = 0

    while iters < MAX_SIMPLEX_ITERS:
        # ── Step 1: Simplex multipliers y = B⁻ᵀ cᴮ ──────────────────────────
        c_B = c_aug[basis]
        y = lu.solve_transpose(c_B)

        # ── Step 2: Reduced costs for non-basic variables ─────────────────────
        non_basic = np.setdiff1d(np.arange(n_aug), basis, assume_unique=True)
        rc = c_aug[non_basic] - A_aug[:, non_basic].T.dot(y)

        # ── Step 3: Optimality check ──────────────────────────────────────────
        if len(rc) == 0 or rc.min() >= -OPTIMALITY_TOL:
            break

        # ── Step 4: Choose entering variable ─────────────────────────────────
        if use_bland or iters > BLAND_RULE_THRESHOLD:
            neg_idx = np.where(rc < -OPTIMALITY_TOL)[0]
            enter_local = neg_idx[np.argmin(non_basic[neg_idx])]
        else:
            enter_local = int(np.argmin(rc))
        enter_col = non_basic[enter_local]

        # ── Step 5: Simplex direction d = B⁻¹ a_s ────────────────────────────
        a_s = np.asarray(A_aug[:, enter_col].todense()).ravel()
        d = lu.solve(a_s)

        # ── Step 6: Current BFS + minimum ratio test ──────────────────────────
        x_B = lu.solve(b_aug)
        pos_mask = d > PIVOT_TOL
        if not pos_mask.any():
            return SolveResult("unbounded", None, -np.inf, iters, "Problem is unbounded.")

        ratios = np.where(pos_mask, x_B / d, np.inf)
        if use_bland or iters > BLAND_RULE_THRESHOLD:
            min_r = ratios.min()
            cands = np.where(np.abs(ratios - min_r) < 1e-12)[0]
            leave_local = int(cands[np.argmin(basis[cands])])
        else:
            leave_local = int(np.argmin(ratios))

        # ── Step 7: Update basis ──────────────────────────────────────────────
        basis[leave_local] = enter_col
        pivot_count_since_refact += 1
        if pivot_count_since_refact >= REFACTORIZE_EVERY:
            try:
                lu.refactorize(_bmat())
            except ValueError:
                return SolveResult("infeasible", None, np.inf, iters, "Basis became singular.")
            pivot_count_since_refact = 0

        iters += 1

    else:
        return SolveResult(
            "iteration_limit", None, np.inf, iters,
            f"Primal simplex hit max iterations ({MAX_SIMPLEX_ITERS})."
        )

    # ── Extract solution ──────────────────────────────────────────────────────
    try:
        lu.refactorize(_bmat())
    except ValueError:
        pass

    x_B = lu.solve(b_aug)

    # Infeasibility check: artificials in basis with non-zero value
    art_in_basis = basis >= art_start
    if np.any(art_in_basis):
        art_vals = x_B[art_in_basis]
        if np.any(np.abs(art_vals) > FEASIBILITY_TOL):
            return SolveResult(
                "infeasible", None, np.inf, iters,
                "Problem is infeasible (artificials remain in basis)."
            )

    # Build full solution
    x_std = np.zeros(n_aug)
    x_std[basis] = np.maximum(0.0, x_B)
    x_shifted = x_std[:n_orig]
    x_orig = x_shifted + lb_finite
    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult("optimal", x_orig, obj_val, iters, basis=basis)



# ── Revised Dual Simplex (warm-start capable) ─────────────────────────────────

def solve_lp_dual(
    problem: Problem,
    basis: Optional[np.ndarray] = None,
    bound_changes: Optional[dict] = None,
) -> SolveResult:
    """
    Solve an LP using the dual revised simplex.

    Starting from a dual-feasible (but possibly primal-infeasible) basis,
    restores primal feasibility by dual pivoting.

    Used in B&B to warm-start after a bound change:
    - Branch adds a bound: x_j ≤ floor(x*_j) or x_j ≥ ceil(x*_j)
    - This preserves dual feasibility (optimal reduced costs unchanged)
    - Dual simplex re-optimises in typically a handful of pivots

    Parameters
    ----------
    problem : Problem
        LP relaxation (integer_mask may be set but is ignored here).
    basis : ndarray of int (m_total,)
        Starting basis column indices. If None, falls back to primal simplex.
    bound_changes : dict {var_idx: (new_lb, new_ub)}
        Bound changes to apply before solving (for B&B warm-start).

    Returns
    -------
    SolveResult
    """
    # Apply bound changes (modify problem in-place copy)
    if bound_changes:
        lb = problem.lb.copy()
        ub = problem.ub.copy()
        for var_idx, (new_lb, new_ub) in bound_changes.items():
            lb[var_idx] = max(lb[var_idx], new_lb) if not np.isinf(new_lb) else lb[var_idx]
            ub[var_idx] = min(ub[var_idx], new_ub) if not np.isinf(new_ub) else ub[var_idx]
        import dataclasses
        problem = dataclasses.replace(problem, lb=lb, ub=ub)

    # If no basis provided, fall back to primal simplex
    if basis is None:
        return solve_lp_revised(problem)

    (c_aug, A_aug, b_aug, basis0, lb_finite,
     n_orig, n_aug, art_start, n_art, BIG_M) = _build_standard_form(problem)

    m = A_aug.shape[0]

    basis = basis.copy().astype(int)

    try:
        lu = SparseLU(A_aug[:, basis])
    except ValueError:
        return solve_lp_revised(problem)

    iters = 0
    pivot_count_since_refact = 0

    while iters < MAX_SIMPLEX_ITERS:
        # Current BFS
        x_B = lu.solve(b_aug)

        # ── Check primal feasibility ──────────────────────────────────────────
        neg_mask = x_B < -FEASIBILITY_TOL
        if not neg_mask.any():
            break  # primal feasible → optimal

        # ── Leaving variable: most infeasible ─────────────────────────────────
        leave_local = int(np.argmin(x_B))

        # ── Dual simplex direction: u = B⁻ᵀ eᵣ ──────────────────────────────
        e_r = np.zeros(m)
        e_r[leave_local] = 1.0
        u = lu.solve_transpose(e_r)

        # ── Dual ratio test: entering variable ───────────────────────────────
        non_basic = np.setdiff1d(np.arange(n_aug), basis, assume_unique=True)

        c_B = c_aug[basis]
        y = lu.solve_transpose(c_B)
        rc = c_aug[non_basic] - A_aug[:, non_basic].T.dot(y)

        a_non = A_aug[:, non_basic]
        pivot_col_coeffs = a_non.T.dot(u)

        neg_pc = pivot_col_coeffs < -PIVOT_TOL
        if not neg_pc.any():
            return SolveResult("infeasible", None, np.inf, iters, "Problem is infeasible (dual).")

        dual_ratios = np.where(neg_pc, rc / (-pivot_col_coeffs), np.inf)
        enter_local = int(np.argmin(dual_ratios))
        enter_col = non_basic[enter_local]

        # ── Pivot ─────────────────────────────────────────────────────────────
        basis[leave_local] = enter_col
        pivot_count_since_refact += 1

        if pivot_count_since_refact >= REFACTORIZE_EVERY:
            try:
                lu.refactorize(A_aug[:, basis])
            except ValueError:
                return solve_lp_revised(problem)
            pivot_count_since_refact = 0

        iters += 1

    else:
        return solve_lp_revised(problem)

    # ── Extract solution ──────────────────────────────────────────────────────
    lu.refactorize(A_aug[:, basis])
    x_B = lu.solve(b_aug)

    x_std = np.zeros(n_aug)
    x_std[basis] = np.maximum(0.0, x_B)
    x_shifted = x_std[:n_orig]
    x_orig = x_shifted + lb_finite
    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult("optimal", x_orig, obj_val, iters, basis=basis)
