"""
solver/milp/gomory.py
---------------------
Gomory Fractional and Mixed-Integer Gomory (MIG) Cutting Plane Generator.

Generates valid cutting planes from the optimal simplex tableau or from
aggregated constraints:
    aᵀx ≤ b

Gomory cuts are derived by decomposing coefficients into integer and fractional
parts. For any tableau row corresponding to a basic integer variable:
    x_Bi + ∑_j ā_ij x_j = b̄_i

where b̄_i is fractional (f_0 = b̄_i - ⌊b̄_i⌋ > 0), the cut separates the
current LP relaxation solution x* while preserving all integer-feasible solutions.

References:
    Gomory, R. E. (1958). "Outline of an algorithm for integer solutions to
    linear programs." Bull. Amer. Math. Soc. 64(5): 275-278.
    Balas, E., Ceria, S., Cornuéjols, G., & Natraj, N. (1996). "Gomory cuts
    revisited." Operations Research Letters, 19(1), 1-9.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import CUT_VIOLATION_MIN, MAX_CUTS_PER_ROUND
from solver.lp.simplex_revised import _build_standard_form
from solver.problem import Problem
from solver.utils.sparse_lu import SparseLU

CutRow = Tuple[np.ndarray, float]  # (a, b) for aᵀx ≤ b


def generate_gomory_cuts(
    problem: Problem,
    x_star: np.ndarray,
    basis: Optional[np.ndarray] = None,
    max_cuts: int = MAX_CUTS_PER_ROUND,
) -> List[CutRow]:
    """
    Generate Gomory fractional and mixed-integer cuts from the LP relaxation.

    Parameters
    ----------
    problem : Problem
        MILP problem instance with integer_mask.
    x_star : np.ndarray
        Optimal solution to the LP relaxation.
    basis : Optional[np.ndarray]
        Basis indices from simplex_revised.
    max_cuts : int
        Maximum number of cuts to return.

    Returns
    -------
    List of (a_cut, b_cut) satisfying a_cutᵀx ≤ b_cut.
    """
    cuts: List[CutRow] = []
    int_mask = problem.integer_mask
    if not int_mask.any():
        return cuts

    # Check if there are fractional integer variables in x_star
    int_vars = np.where(int_mask)[0]
    fracs = np.abs(x_star[int_vars] - np.round(x_star[int_vars]))
    frac_vars = int_vars[fracs > 1e-4]
    if len(frac_vars) == 0:
        return cuts

    # 1. Tableau-based Gomory cut generation if basis is available
    if basis is not None:
        try:
            (
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
            ) = _build_standard_form(problem)

            m = A_aug.shape[0]
            basis_clean = np.clip(basis.copy().astype(int), 0, n_aug - 1)
            lu = SparseLU(A_aug[:, basis_clean])
            x_B = lu.solve(b_aug)

            # Map which basic rows correspond to original integer variables
            m_ub = problem.n_ineq
            ub_finite_cols = np.where(np.isfinite(problem.ub))[0]

            for row_i in range(m):
                col = basis_clean[row_i]
                if col < n_orig and int_mask[col]:
                    val = x_B[row_i]
                    f0 = val - np.floor(val)
                    if 1e-4 < f0 < 1.0 - 1e-4:
                        # Tableau row: u = B⁻ᵀ e_i, row = uᵀ A_aug
                        e_i = np.zeros(m)
                        e_i[row_i] = 1.0
                        u = lu.solve_transpose(e_i)
                        a_bar = np.asarray(A_aug.T.dot(u)).ravel()

                        # Standard Gomory / MIG cut coefficients for standard form:
                        # α_k on standard variables x_std >= 0
                        alpha = np.zeros(n_aug)
                        denom = 1.0 - f0

                        for k in range(n_aug):
                            a_k = a_bar[k]
                            if abs(a_k) < 1e-12:
                                continue
                            is_int = (k < n_orig and int_mask[k])
                            if is_int:
                                fk = a_k - np.floor(a_k)
                                if fk <= f0:
                                    alpha[k] = fk
                                else:
                                    alpha[k] = fk - (fk - f0) / denom if denom > 1e-9 else fk
                            else:
                                # Continuous variable
                                if a_k >= 0:
                                    alpha[k] = a_k
                                else:
                                    alpha[k] = -a_k * (f0 / denom) if denom > 1e-9 else 0.0

                        # Map α_k back to original variables x:
                        # x_std[j] = x[j] - lb_finite[j] for j < n_orig
                        # slack_r = b_ub[r] - A_ub[r] @ x for r < m_ub
                        # slack_ub = ub[j] - x[j] for upper bounds
                        a_cut = np.zeros(n_orig)
                        rhs_const = f0

                        # Original variables
                        for j in range(n_orig):
                            a_cut[j] += alpha[j]
                            rhs_const += alpha[j] * lb_finite[j]

                        # Inequality slacks
                        if m_ub > 0:
                            A_ub_dense = problem.A_ub.toarray()
                            for r in range(m_ub):
                                alpha_slack = alpha[n_orig + r]
                                if abs(alpha_slack) > 1e-12:
                                    a_cut -= alpha_slack * A_ub_dense[r]
                                    rhs_const -= alpha_slack * problem.b_ub[r]

                        # Upper bound slacks
                        for k, j_ub in enumerate(ub_finite_cols):
                            alpha_ub = alpha[n_orig + m_ub + k]
                            if abs(alpha_ub) > 1e-12:
                                a_cut[j_ub] -= alpha_ub
                                rhs_const -= alpha_ub * problem.ub[j_ub]

                        # Cut is a_cutᵀ x >= rhs_const => -a_cutᵀ x <= -rhs_const
                        cut_a = -a_cut
                        cut_b = -rhs_const

                        violation = float(cut_a @ x_star) - cut_b
                        if violation > CUT_VIOLATION_MIN:
                            cuts.append((cut_a, cut_b))
                            if len(cuts) >= max_cuts:
                                return cuts
        except Exception:
            pass

    # If tableau cuts found, return them
    if cuts:
        return cuts

    # 2. Row-based Chvátal-Gomory / Fractional cut fallback
    if len(cuts) < max_cuts and problem.n_ineq > 0:
        A_ub = problem.A_ub.toarray()
        b_ub = problem.b_ub

        for i in range(problem.n_ineq):
            a_row = A_ub[i]
            b_i = b_ub[i]
            f0 = b_i - np.floor(b_i)

            if 1e-4 < f0 < 1.0 - 1e-4:
                cut_a = np.zeros(problem.n_vars)
                denom = 1.0 - f0

                for j in range(problem.n_vars):
                    a_ij = a_row[j]
                    if int_mask[j]:
                        f_a = a_ij - np.floor(a_ij)
                        if f_a <= f0:
                            cut_a[j] = np.floor(a_ij)
                        else:
                            cut_a[j] = np.floor(a_ij) + (f_a - f0) / denom if denom > 1e-9 else np.floor(a_ij)
                    else:
                        if a_ij > 0:
                            cut_a[j] = a_ij / denom if denom > 1e-9 else a_ij
                        else:
                            cut_a[j] = 0.0

                cut_b = np.floor(b_i)

                violation = float(cut_a @ x_star) - cut_b
                if violation > CUT_VIOLATION_MIN:
                    cuts.append((cut_a, cut_b))
                    if len(cuts) >= max_cuts:
                        break

    return cuts
