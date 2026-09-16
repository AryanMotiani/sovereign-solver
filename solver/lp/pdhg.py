"""
solver/lp/pdhg.py
-----------------
Restarted Primal-Dual Hybrid Gradient (PDHG) for large-scale LP.

Standard form solved:
    min   cᵀx
    s.t.  Ax = b,   x ≥ 0

Algorithm — Chambolle-Pock with adaptive restarts (PDLP approach):
    τ   = step size for primal (x) update
    σ   = step size for dual   (y) update
    τσ ‖A‖² ≤ 1   (stability condition)

Iteration:
    x̂  = prox_{τ f}(x - τ Aᵀ y)   = clip(x - τ Aᵀ y - τc, 0, ∞)
    ȳ  = y + σ A (2x̂ - x)
    x  ← x̂
    y  ← ȳ

Adaptive restart (Applegate et al. 2021):
    - Restart when the current point is "worse" in a running average sense.
    - Period: check every PDHG_RESTART_PERIOD iterations.

Convergence check:
    primal residual:  ‖Ax - b‖ / (1 + ‖b‖)  < ε
    dual residual:    ‖Aᵀy + s - c‖ / (1 + ‖c‖) < ε  (s = c - Aᵀy clipped to ≥0)
    gap:              |cᵀx - bᵀy| / (1 + |cᵀx|) < ε

References:
    Applegate et al. (2021) "Practical Large-Scale Linear Programming using PDLP",
    NeurIPS 2021. arXiv:2106.04756.
    Chambolle & Pock (2011) JMIV.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver.config import (
    FEASIBILITY_TOL,
    MAX_PDHG_ITERS,
    OPTIMALITY_TOL,
    PDHG_RESTART_PERIOD,
    PDHG_TOL,
)
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem


# ── Standard-form conversion ──────────────────────────────────────────────────

def _to_standard_form_pdhg(problem: Problem) -> tuple:
    """
    Convert LP to equality standard form: min cᵀx  s.t. Ax=b, x≥0.
    Same approach as IPM: variable shift + slack variables.

    For finite upper bounds ub[j], adds explicit constraint row:
        x'_j <= ub[j] - lb[j]  (shifted variable)

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

def _compute_step_sizes(A: sp.csr_matrix) -> tuple:
    """
    Compute per-variable primal step sizes (τ_j) and per-constraint dual
    step sizes (σ_i) using diagonal preconditioning (Pock & Chambolle 2011):

        τ_j = 1 / (‖A_j‖_1 + 1)   (column j scaled by sum of |a_ij|)
        σ_i = 1 / (‖A_i‖_1 + 1)   (row i scaled by sum of |a_ij|)

    This is equivalent to the Sinkhorn-Knopp-style diagonal preconditioner
    and gives much faster practical convergence than a uniform step.
    Returns (tau, sigma) — both are 1-D arrays.
    """
    col_norms = np.array(np.abs(A).sum(axis=0)).ravel() + 1.0
    row_norms = np.array(np.abs(A).sum(axis=1)).ravel() + 1.0
    tau   = 1.0 / col_norms    # shape (n,)
    sigma = 1.0 / row_norms    # shape (m,)
    return tau, sigma


# ── Restart logic ─────────────────────────────────────────────────────────────

def _should_restart(
    x: np.ndarray,
    y: np.ndarray,
    x_avg: np.ndarray,
    y_avg: np.ndarray,
    A: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    """
    Restart if the running average point has lower "potential" than current.
    Uses normalized primal + dual residual as the potential.
    """
    norm_b = max(1.0, np.linalg.norm(b))
    norm_c = max(1.0, np.linalg.norm(c))

    def _potential(xp, yp):
        rp = np.linalg.norm(b - A.dot(xp)) / norm_b
        s = np.maximum(c - A.T.dot(yp), 0.0)
        rd = np.linalg.norm(c - A.T.dot(yp) - s) / norm_c
        return rp + rd

    return _potential(x_avg, y_avg) < _potential(x, y)


# ── Main PDHG solver ──────────────────────────────────────────────────────────

def solve_lp_pdhg(problem: Problem) -> SolveResult:
    """
    Solve an LP using restarted PDHG (Chambolle-Pock with adaptive restarts).

    Best suited for very large, sparse LPs where factorization-based methods
    are too expensive. Convergence is first-order (typically O(1/k)).

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

    # Running sums for averaging (used for restart and convergence)
    x_sum = x.copy()
    y_sum = y.copy()

    # Precompute A.T for repeated use
    AT = A.T.tocsr()

    status = "iteration_limit"
    iters = 0
    restart_count = 0

    for iters in range(1, MAX_PDHG_ITERS + 1):
        # ── Primal update: x_new = clip(x - tau*(A'y + c), 0, inf) ───────────
        x_new = np.maximum(x - tau * (AT.dot(y) + c), 0.0)

        # ── Dual update (extrapolated, per-row sigma) ─────────────────────────
        y = y + sigma * A.dot(2.0 * x_new - x)

        x = x_new
        x_sum += x
        y_sum += y

        # ── Check convergence every PDHG_RESTART_PERIOD iters ─────────────
        if iters % PDHG_RESTART_PERIOD == 0:
            x_avg = x_sum / iters
            y_avg = y_sum / iters

            # Primal residual: ‖Ax - b‖
            r_p = b - A.dot(x_avg)
            # Dual slack:  s = max(c - Aᵀy, 0)
            s = np.maximum(c - AT.dot(y_avg), 0.0)
            r_d = c - AT.dot(y_avg) - s

            norm_b = max(1.0, np.linalg.norm(b))
            norm_c = max(1.0, np.linalg.norm(c))

            primal_res = np.linalg.norm(r_p) / norm_b
            dual_res   = np.linalg.norm(r_d) / norm_c
            gap        = abs(float(c @ x_avg) - float(b @ y_avg)) / (
                1.0 + abs(float(c @ x_avg))
            )

            if max(primal_res, dual_res, gap) < PDHG_TOL:
                x = x_avg
                y = y_avg
                status = "optimal"
                break

            # ── Adaptive restart ───────────────────────────────────────────
            if _should_restart(x, y, x_avg, y_avg, A, b, c):
                # Restart from average
                x = x_avg.copy()
                y = y_avg.copy()
                x_sum = x.copy()
                y_sum = y.copy()
                restart_count += 1

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
