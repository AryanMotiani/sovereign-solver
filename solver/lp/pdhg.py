"""
solver/lp/pdhg.py
-----------------
Restarted Primal-Dual Hybrid Gradient (PDHG) for large-scale LP.

Standard form solved:
    min   cᵀx
    s.t.  Ax = b,   x ≥ 0

Lagrangian saddle-point formulation:
    L(x, y) = cᵀx + yᵀ(b - Ax)    with x ≥ 0, y free

Algorithm — Chambolle-Pock with adaptive restarts (PDLP approach):
    Diagonal preconditioning (Pock & Chambolle 2011):
        τ_j = 0.999 / ∑_i |A_ij|   (primal column step size)
        σ_i = 0.999 / ∑_j |A_ij|   (dual row step size)
    Condition: ‖Σ^{1/2} A T^{1/2}‖ ≤ 1

Iteration:
    x_{k+1} = max(0, x_k - τ (c - Aᵀ y_k))
    x̄       = 2 x_{k+1} - x_k
    y_{k+1} = y_k + σ (b - A x̄)

Adaptive restart (Applegate et al. 2021 PDLP):
    Accumulate ergodic running average (x_avg, y_avg).
    Evaluate normalized KKT error = max(primal_res, dual_res, duality_gap).
    If KKT error decreases by factor ≥ 2 (sufficient decrease), restart from
    the average point and reset the averaging window.

Convergence check:
    primal residual:  ‖Ax - b‖ / (1 + ‖b‖) < ε
    dual residual:    ‖c - Aᵀy - s‖ / (1 + ‖c‖) < ε  (where s = max(c - Aᵀy, 0))
    duality gap:      |cᵀx - bᵀy| / (1 + |cᵀx| + |bᵀy|) < ε

References:
    Applegate et al. (2021) "Practical Large-Scale Linear Programming using PDLP",
    NeurIPS 2021. arXiv:2106.04756.
    Chambolle & Pock (2011) "A First-Order Primal-Dual Algorithm for Convex Problems
    with Applications to Imaging", JMIV.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    FEASIBILITY_TOL,
    MAX_PDHG_ITERS,
    PDHG_OVER_RELAXATION,
    PDHG_RESTART_PERIOD,
    PDHG_TOL,
)
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem


# ── Standard-form conversion ──────────────────────────────────────────────────

def _to_standard_form_pdhg(problem: Problem) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Convert LP to equality standard form: min cᵀx  s.t. Ax=b, x≥0.
    Handles variable shift, slack variables, and finite upper bounds.
    Returns (c, A_csr, b, lb_orig, n_orig).
    """
    n_orig = problem.n_vars
    m_ub = problem.n_ineq
    m_eq = problem.n_eq

    lb = problem.lb.copy()
    lb_finite = np.where(np.isfinite(lb), lb, 0.0)

    b_ub_s = problem.b_ub - problem.A_ub.dot(lb_finite)
    b_eq_s = problem.b_eq - problem.A_eq.dot(lb_finite)

    # ── Add upper-bound rows for finite ub[j] ─────────────────────────────────
    ub = problem.ub.copy()
    ub_shift = ub - lb_finite
    finite_ub_mask = np.isfinite(ub) & np.isfinite(lb_finite)
    finite_ub_cols = np.where(finite_ub_mask)[0]
    m_ub_extra = len(finite_ub_cols)

    if m_ub_extra > 0:
        rows = np.arange(m_ub_extra)
        cols = finite_ub_cols
        data = np.ones(m_ub_extra)
        A_ub_extra = sp.csr_matrix((data, (rows, cols)), shape=(m_ub_extra, n_orig))
        b_ub_extra = ub_shift[finite_ub_cols]
        A_ub_combined = sp.vstack([problem.A_ub, A_ub_extra], format="csr")
        b_ub_combined = np.concatenate([b_ub_s, b_ub_extra])
        m_ub_total = m_ub + m_ub_extra
    else:
        A_ub_combined = problem.A_ub
        b_ub_combined = b_ub_s
        m_ub_total = m_ub

    n_slacks = m_ub_total
    n_full = n_orig + n_slacks

    if m_ub_total > 0:
        row_ub = sp.hstack([A_ub_combined, sp.eye(m_ub_total, format="csr")], format="csr")
    else:
        row_ub = sp.csr_matrix((0, n_full))

    if m_eq > 0:
        z = sp.csr_matrix((m_eq, n_slacks))
        row_eq = sp.hstack([problem.A_eq, z], format="csr")
    else:
        row_eq = sp.csr_matrix((0, n_full))

    A = sp.vstack([row_ub, row_eq], format="csr")
    b = np.concatenate([b_ub_combined, b_eq_s])

    # Handle negative RHS by row negation
    neg = b < -FEASIBILITY_TOL
    if neg.any():
        A_lil = A.tolil()
        A_lil[neg, :] *= -1
        A = A_lil.tocsr()
        b[neg] *= -1

    c = np.concatenate([problem.c.copy(), np.zeros(n_slacks)])
    return c, A, b, lb_finite, n_orig



# ── Step-size computation ──────────────────────────────────────────────────────

def _compute_step_sizes(A: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-variable primal step sizes (τ_j) and per-constraint dual
    step sizes (σ_i) using diagonal preconditioning (Pock & Chambolle 2011).

    τ_j = 0.999 / max(∑_i |A_ij|, 1e-4)
    σ_i = 0.999 / max(∑_j |A_ij|, 1e-4)
    """
    col_norms = np.array(np.abs(A).sum(axis=0)).ravel()
    row_norms = np.array(np.abs(A).sum(axis=1)).ravel()

    col_norms = np.maximum(col_norms, 1e-4)
    row_norms = np.maximum(row_norms, 1e-4)

    tau = 0.999 / col_norms
    sigma = 0.999 / row_norms
    return tau, sigma


# ── Main PDHG solver ──────────────────────────────────────────────────────────

def solve_lp_pdhg(
    problem: Problem,
    max_iters: int = MAX_PDHG_ITERS,
    tol: float = PDHG_TOL,
) -> SolveResult:
    """
    Solve an LP using restarted PDHG (Chambolle-Pock with adaptive restarts).

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    SolveResult
    """
    c, A, b, lb_orig, n_orig = _to_standard_form_pdhg(problem)
    m, n = A.shape

    if m == 0 or n == 0:
        return SolveResult("optimal", np.zeros(problem.n_vars), 0.0, 0)

    # ── Initialization ──────────────────────────────────────────────────────
    x = np.zeros(n)
    y = np.zeros(m)

    tau, sigma = _compute_step_sizes(A)
    AT = A.T.tocsr()

    norm_b = max(1.0, float(np.linalg.norm(b)))
    norm_c = max(1.0, float(np.linalg.norm(c)))

    def _kkt_residuals(xp: np.ndarray, yp: np.ndarray) -> Tuple[float, float, float, float]:
        # Primal residual: ‖Ax - b‖ / (1 + ‖b‖)
        rp = float(np.linalg.norm(A.dot(xp) - b)) / norm_b
        # Dual slack: s = max(c - Aᵀy, 0), dual residual: ‖c - Aᵀy - s‖ / (1 + ‖c‖)
        A_T_y = AT.dot(yp)
        s = np.maximum(c - A_T_y, 0.0)
        rd = float(np.linalg.norm(c - A_T_y - s)) / norm_c
        # Relative duality gap: |cᵀx - bᵀy| / (1 + |cᵀx| + |bᵀy|)
        p_obj = float(c @ xp)
        d_obj = float(b @ yp)
        gap = abs(p_obj - d_obj) / (1.0 + abs(p_obj) + abs(d_obj))
        max_err = max(rp, rd, gap)
        return max_err, rp, rd, gap

    # Running sums for ergodic averaging
    x_sum = np.zeros(n)
    y_sum = np.zeros(m)
    window_len = 0

    current_kkt, _, _, _ = _kkt_residuals(x, y)
    status = "iteration_limit"
    iters = 0

    check_interval = max(10, min(PDHG_RESTART_PERIOD, 100))

    for iters in range(1, max_iters + 1):
        # ── Primal update: x_{k+1} = max(0, x_k - τ (c - Aᵀ y_k)) ──────────
        x_new = np.maximum(x - tau * (c - AT.dot(y)), 0.0)

        # ── Extrapolation: x̄ = 2 x_{k+1} - x_k ─────────────────────────────
        x_bar = 2.0 * x_new - x

        # ── Dual update: y_{k+1} = y_k + σ (b - A x̄) ───────────────────────
        y = y + sigma * (b - A.dot(x_bar))

        x = x_new
        x_sum += x
        y_sum += y
        window_len += 1

        # ── Check convergence and restart periodically ───────────────────────
        if window_len >= check_interval and window_len % check_interval == 0:
            x_avg = x_sum / window_len
            y_avg = y_sum / window_len

            err, rp, rd, gap = _kkt_residuals(x_avg, y_avg)

            if err < tol:
                x = x_avg
                y = y_avg
                status = "optimal"
                break

            # Adaptive restart: sufficient decrease in KKT error
            if err < 0.5 * current_kkt:
                x = x_avg.copy()
                y = y_avg.copy()
                x_sum.fill(0.0)
                y_sum.fill(0.0)
                window_len = 0
                current_kkt = err

    # ── Extract solution ──────────────────────────────────────────────────────
    x_shifted = x[:n_orig]
    x_orig = x_shifted + lb_orig
    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult(status, x_orig, obj_val, iters)
