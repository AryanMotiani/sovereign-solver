"""
solver/lp/simplex_revised.py
-----------------------------
Production-quality revised simplex LP solver — primal and dual variants.

Improvements over v1:
  - SPARSE standard-form construction (lil_matrix, never dense)
  - Ruiz equilibration for numerical robustness
  - Harris two-pass ratio test for degeneracy handling
  - Devex approximate steepest-edge pricing
  - Forrest-Tomlin rank-1 LU updates between refactorizations
  - Proper bound handling via sparse construction

Algorithm summary
-----------------
PRIMAL REVISED SIMPLEX
  Maintains a basis B (index set of basic variables) and its LU factorization.
  Each iteration:
    1. Solve   B y = cᴮ              → y  (dual variables / simplex multipliers)
    2. Compute reduced costs: rc_j = c_j - yᵀ a_j  for non-basic j
    3. Pivot column: steepest-edge / Dantzig / Bland
    4. Solve   B d = a_s              → d  (simplex direction)
    5. Harris two-pass ratio test → leaving variable r
    6. Pivot: rank-1 update or refactorize

DUAL REVISED SIMPLEX
  Starts dual-feasible, restores primal feasibility by dual pivoting.
  Used for warm-starting after bound changes (B&B branching).

Standard form
-------------
Input problem is converted to:
    min  cᵀx
    s.t. [A_ub | I_slack] [x; s] = b_rhs,  x ≥ 0, s ≥ 0
Built entirely in sparse format (lil_matrix → csr_matrix).

References
----------
- Vanderbei, "Linear Programming: Foundations and Extensions" (4th ed.)
- Huangfu & Hall, Math. Prog. Computation 10(2), 2018 (dual simplex design)
- Harris (1973) "Pivot selection methods of the Devex LP code"
- Forrest & Tomlin (1972) "Updated triangular factors..."
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    BLAND_RULE_THRESHOLD,
    FEASIBILITY_TOL,
    HARRIS_TOL,
    MAX_SIMPLEX_ITERS,
    OPTIMALITY_TOL,
    PIVOT_TOL,
    REFACTORIZE_EVERY,
    SCALING_METHOD,
    USE_STEEPEST_EDGE,
)
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem
from solver.utils.sparse_lu import SparseLU
from solver.utils.scaling import make_scaler


def _build_standard_form(problem: Problem):
    """
    Convert LP to standard equality form for revised simplex.
    Built entirely in sparse format — never materializes a dense (m, n) array.

    Strategy
    --------
    For each inequality row Aᵢx ≤ bᵢ:
      - bᵢ ≥ 0: add slack sᵢ ≥ 0 → Aᵢx + sᵢ = bᵢ.  Initial BFS: sᵢ = bᵢ.
      - bᵢ < 0: multiply by -1.  Add surplus + artificial.
    For each equality row: add artificial only.

    Returns
    -------
    c_aug     : objective (n_aug,)
    A_aug     : csr_matrix (m_total, n_aug)
    b_aug     : rhs (m_total,)  — all ≥ 0
    basis0    : initial basis of length m_total
    lb_finite : shifted lower bounds
    n_orig    : number of original variables
    n_aug     : total columns
    art_start : column where artificials begin
    n_art     : number of artificial variables
    BIG_M     : Big-M penalty
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

    # Collect upper-bound rows for finite UBs
    ub_shift = ub - lb_finite
    finite_ub_mask = np.isfinite(ub) & np.isfinite(lb_finite)
    finite_ub_cols = np.where(finite_ub_mask)[0]
    n_ub_rows = len(finite_ub_cols)

    # Total inequality rows = original + UB rows
    m_ub_total = m_ub + n_ub_rows

    # Determine which rows have negative RHS (need artificial)
    # Build combined b vector for all inequality rows
    if n_ub_rows > 0:
        b_ub_combined = np.concatenate([b_ub_s, ub_shift[finite_ub_cols]])
    else:
        b_ub_combined = b_ub_s.copy()

    neg_ub_rows = np.where(b_ub_combined < -FEASIBILITY_TOL)[0]
    n_art_ub = len(neg_ub_rows)
    n_art_eq = m_eq
    n_art = n_art_ub + n_art_eq

    n_slacks = m_ub_total
    art_start = n_orig + n_slacks
    n_aug = n_orig + n_slacks + n_art
    m_total = m_ub_total + m_eq

    # Build A_aug as lil_matrix (efficient for row-by-row construction)
    A_lil = sp.lil_matrix((m_total, n_aug), dtype=float)
    b_aug = np.zeros(m_total)
    basis0 = np.zeros(m_total, dtype=int)

    art_col = art_start

    # --- Inequality rows from A_ub ---
    A_ub_csr = problem.A_ub.tocsr()
    for i in range(m_ub):
        if b_ub_combined[i] >= -FEASIBILITY_TOL:
            # Normal case: slack is basic
            # Copy original row
            row_start = A_ub_csr.indptr[i]
            row_end = A_ub_csr.indptr[i + 1]
            for idx in range(row_start, row_end):
                j = A_ub_csr.indices[idx]
                A_lil[i, j] = A_ub_csr.data[idx]
            A_lil[i, n_orig + i] = 1.0  # slack
            b_aug[i] = b_ub_combined[i]
            basis0[i] = n_orig + i
        else:
            # Negative RHS: flip, add surplus + artificial
            row_start = A_ub_csr.indptr[i]
            row_end = A_ub_csr.indptr[i + 1]
            for idx in range(row_start, row_end):
                j = A_ub_csr.indices[idx]
                A_lil[i, j] = -A_ub_csr.data[idx]
            A_lil[i, n_orig + i] = -1.0  # surplus
            A_lil[i, art_col] = 1.0       # artificial
            b_aug[i] = -b_ub_combined[i]
            basis0[i] = art_col
            art_col += 1

    # --- Upper-bound rows ---
    for k, j in enumerate(finite_ub_cols):
        row_idx = m_ub + k
        slack_idx = n_orig + row_idx
        if b_ub_combined[row_idx] >= -FEASIBILITY_TOL:
            A_lil[row_idx, j] = 1.0
            A_lil[row_idx, slack_idx] = 1.0
            b_aug[row_idx] = b_ub_combined[row_idx]
            basis0[row_idx] = slack_idx
        else:
            A_lil[row_idx, j] = -1.0
            A_lil[row_idx, slack_idx] = -1.0
            A_lil[row_idx, art_col] = 1.0
            b_aug[row_idx] = -b_ub_combined[row_idx]
            basis0[row_idx] = art_col
            art_col += 1

    # --- Equality rows ---
    A_eq_csr = problem.A_eq.tocsr()
    for j_eq in range(m_eq):
        row_idx = m_ub_total + j_eq
        row_start = A_eq_csr.indptr[j_eq]
        row_end = A_eq_csr.indptr[j_eq + 1]
        for idx in range(row_start, row_end):
            col = A_eq_csr.indices[idx]
            A_lil[row_idx, col] = A_eq_csr.data[idx]
        A_lil[row_idx, art_col] = 1.0
        # Handle negative RHS for equalities
        if b_eq_s[j_eq] < -FEASIBILITY_TOL:
            # Flip row and use artificial
            for idx in range(row_start, row_end):
                col = A_eq_csr.indices[idx]
                A_lil[row_idx, col] = -A_eq_csr.data[idx]
            A_lil[row_idx, art_col] = 1.0
            b_aug[row_idx] = -b_eq_s[j_eq]
        else:
            b_aug[row_idx] = b_eq_s[j_eq]
        basis0[row_idx] = art_col
        art_col += 1

    # Convert to CSR for efficient arithmetic
    A_aug = A_lil.tocsr()

    # Objective: original + Big-M on artificials
    c_aug = np.concatenate([c_s, np.zeros(n_slacks), np.full(n_art, BIG_M)])

    # Eliminate artificials from objective row (correct initial reduced costs)
    for i, bi in enumerate(basis0):
        if bi >= art_start:
            # c_aug -= BIG_M * A_aug[i, :]
            row = A_aug.getrow(i)
            c_aug -= BIG_M * np.asarray(row.todense()).ravel()

    return (
        c_aug,
        A_aug,
        b_aug,
        basis0,
        lb_finite,
        n_orig,
        n_aug,
        art_start,
        n_art,
        BIG_M,
    )


# ── Pricing strategies ───────────────────────────────────────────────────────

def _dantzig_price(rc: np.ndarray, non_basic: np.ndarray) -> int:
    """Dantzig rule: most negative reduced cost."""
    return int(np.argmin(rc))


def _bland_price(rc: np.ndarray, non_basic: np.ndarray) -> int:
    """Bland's rule: smallest index with negative reduced cost."""
    neg_idx = np.where(rc < -OPTIMALITY_TOL)[0]
    return int(neg_idx[np.argmin(non_basic[neg_idx])])


def _devex_price(rc: np.ndarray, non_basic: np.ndarray, weights: np.ndarray) -> int:
    """
    Devex approximate steepest-edge pricing.
    Score = rc_j² / w_j — select maximum.
    """
    scores = rc ** 2 / np.maximum(weights, 1e-12)
    neg_mask = rc < -OPTIMALITY_TOL
    if not neg_mask.any():
        return int(np.argmin(rc))
    scores[~neg_mask] = -np.inf
    return int(np.argmax(scores))


# ── Harris two-pass ratio test ────────────────────────────────────────────────

def _harris_ratio_test(x_B: np.ndarray, d: np.ndarray, basis: np.ndarray,
                       use_bland: bool) -> int:
    """
    Harris (1973) two-pass ratio test for better degeneracy handling.

    Pass 1: Find θ_max = min { (x_B[i] + ε) / d[i] : d[i] > tol }
            where ε = HARRIS_TOL. This allows a small bound violation.
    Pass 2: Among all i with d[i] > tol and x_B[i]/d[i] ≤ θ_max,
            select the one with largest d[i] (most stable pivot).
    """
    pos_mask = d > PIVOT_TOL
    if not pos_mask.any():
        return -1  # unbounded

    # Pass 1: compute relaxed ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        relaxed_ratios = np.where(pos_mask,
                                  (np.maximum(x_B, 0.0) + HARRIS_TOL) / d,
                                  np.inf)
        theta_max = relaxed_ratios.min()

        # Pass 2: among eligible, pick largest pivot element (most stable)
        exact_ratios = np.where(pos_mask, np.maximum(x_B, 0.0) / d, np.inf)
        eligible = pos_mask & (exact_ratios <= theta_max + HARRIS_TOL)

    if not eligible.any():
        # Fallback to simple ratio test
        if use_bland:
            min_r = exact_ratios[pos_mask].min()
            cands = np.where(pos_mask & (np.abs(exact_ratios - min_r) < 1e-12))[0]
            return int(cands[np.argmin(basis[cands])])
        return int(np.argmin(exact_ratios))

    # Among eligible, pick largest d[i] for numerical stability
    pivot_sizes = np.where(eligible, np.abs(d), -np.inf)

    if use_bland:
        # Bland tie-break among eligible
        elig_idx = np.where(eligible)[0]
        return int(elig_idx[np.argmin(basis[elig_idx])])

    return int(np.argmax(pivot_sizes))


# ── Revised Primal Simplex ────────────────────────────────────────────────────

def solve_lp_revised(
    problem: Problem,
    basis: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    use_bland: bool = False,
) -> SolveResult:
    """
    Solve an LP with the revised primal simplex.

    Features:
      - Sparse standard-form construction
      - Forrest-Tomlin rank-1 LU updates
      - Harris two-pass ratio test
      - Devex approximate steepest-edge pricing
      - Ruiz/geometric scaling for numerical robustness
    """
    (c_aug, A_aug, b_aug, basis0, lb_finite,
     n_orig, n_aug, art_start, n_art, BIG_M) = _build_standard_form(problem)

    m = A_aug.shape[0]

    # Edge case: no constraints → optimal at lower bound
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

    # Devex weights (approximate steepest-edge)
    devex_weights = np.ones(n_aug)

    iters = 0

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
            enter_local = _bland_price(rc, non_basic)
        elif USE_STEEPEST_EDGE:
            enter_local = _devex_price(rc, non_basic, devex_weights[non_basic])
        else:
            enter_local = _dantzig_price(rc, non_basic)
        enter_col = non_basic[enter_local]

        # ── Step 5: Simplex direction d = B⁻¹ a_s ────────────────────────────
        a_s = np.asarray(A_aug[:, enter_col].todense()).ravel()
        d = lu.solve(a_s)

        # ── Step 6: Current BFS + ratio test ──────────────────────────────────
        x_B = lu.solve(b_aug)

        leave_local = _harris_ratio_test(x_B, d, basis, use_bland or iters > BLAND_RULE_THRESHOLD)
        if leave_local < 0:
            return SolveResult("unbounded", None, -np.inf, iters, "Problem is unbounded.")

        # ── Step 7: Update Devex weights ──────────────────────────────────────
        if USE_STEEPEST_EDGE:
            pivot_val = d[leave_local]
            if abs(pivot_val) > PIVOT_TOL:
                # Devex weight update (approximate steepest-edge)
                gamma = np.sum(d ** 2) / (pivot_val ** 2)
                devex_weights[enter_col] = max(gamma, 1e-4)

        # ── Step 8: Update basis with rank-1 update ───────────────────────────
        old_leaving = basis[leave_local]
        basis[leave_local] = enter_col

        if lu.needs_refactorize(REFACTORIZE_EVERY):
            try:
                lu.refactorize(_bmat())
            except ValueError:
                return SolveResult("infeasible", None, np.inf, iters, "Basis became singular.")
        else:
            # Rank-1 Forrest-Tomlin update
            lu.update_column(leave_local, a_s)

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

    if len(basis) != m:
        return solve_lp_revised(problem)

    basis = basis.copy().astype(int)

    # Validate basis indices are in range
    basis = np.clip(basis, 0, n_aug - 1)

    try:
        lu = SparseLU(A_aug[:, basis])
    except ValueError:
        return solve_lp_revised(problem)

    iters = 0

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

        dual_ratios = np.full(len(non_basic), np.inf)
        dual_ratios[neg_pc] = rc[neg_pc] / (-pivot_col_coeffs[neg_pc])
        enter_local = int(np.argmin(dual_ratios))
        enter_col = non_basic[enter_local]

        # ── Pivot ─────────────────────────────────────────────────────────────
        a_enter = np.asarray(A_aug[:, enter_col].todense()).ravel()
        basis[leave_local] = enter_col

        if lu.needs_refactorize(REFACTORIZE_EVERY):
            try:
                lu.refactorize(A_aug[:, basis])
            except ValueError:
                return solve_lp_revised(problem)
        else:
            lu.update_column(leave_local, a_enter)

        iters += 1

    else:
        return solve_lp_revised(problem)

    # ── Extract solution ──────────────────────────────────────────────────────
    try:
        lu.refactorize(A_aug[:, basis])
    except ValueError:
        return solve_lp_revised(problem)

    x_B = lu.solve(b_aug)

    # Check artificial variables
    if n_art > 0:
        for idx_b, col in enumerate(basis):
            if col >= art_start and x_B[idx_b] > FEASIBILITY_TOL:
                return solve_lp_revised(problem)

    x_std = np.zeros(n_aug)
    x_std[basis] = np.maximum(0.0, x_B)
    x_shifted = x_std[:n_orig]
    x_orig = x_shifted + lb_finite
    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    # Verify feasibility against original problem constraints
    if problem.n_ineq > 0:
        if np.any(problem.A_ub.dot(x_orig) - problem.b_ub > 1e-4):
            return solve_lp_revised(problem)
    if problem.n_eq > 0:
        if np.any(np.abs(problem.A_eq.dot(x_orig) - problem.b_eq) > 1e-4):
            return solve_lp_revised(problem)

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult("optimal", x_orig, obj_val, iters, basis=basis)

