"""
solver/lp/interior_point.py
-----------------------------
Mehrotra Predictor-Corrector Interior-Point Method for LP (from scratch).

Standard form:
    min   cᵀx
    s.t.  Ax = b,   x ≥ 0

Algorithm (Mehrotra 1992, Wright 1997 §6.3):
--------------------------------------------
At each iteration, solve for the Newton direction from the KKT conditions:

    A Δx         = r_p = b - Ax                   (primal feasibility)
    Aᵀ Δy + Δs  = r_d = c - Aᵀy - s              (dual feasibility)
    S Δx + X Δs = r_c = -XSe + σμe - ΔxₐΔsₐ      (complementarity)

Eliminating Δs = r_d - Aᵀ Δy and substituting into the complementarity:

    (A D Aᵀ) Δy = r_p + A D r_d + A (r_c/s)

where D = X S⁻¹.  Then Δx = D (r_d - Aᵀ Δy + r_c/x).

Mehrotra's two-phase (predictor-corrector):
  Phase 1 (Predictor): σ=0, r_c = -XSe.
  Phase 2 (Corrector): σ = (μ_aff/μ)³,
                        r_c = -XSe + σμe - ΔxₐΔsₐ.

References:
    Wright (1997) "Primal-Dual Interior-Point Methods", SIAM, §6.3
    Mehrotra (1992) SIAM J. Optim. 2(4):575-601.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import FEASIBILITY_TOL, MAX_IPM_ITERS, OPTIMALITY_TOL
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem


# ── Standard-form conversion ──────────────────────────────────────────────────

def _to_standard_form_ipm(problem: Problem) -> tuple:
    """
    Convert LP to equality standard form:  min cᵀx  s.t. Ax=b, x≥0.
    Handles negative-RHS rows correctly.
    Returns (c, A_csr, b, lb_orig, n_orig).
    """
    n_orig = problem.n_vars
    m_ub = problem.n_ineq
    m_eq = problem.n_eq

    lb = problem.lb.copy()
    lb_finite = np.where(np.isfinite(lb), lb, 0.0)

    b_ub_s = problem.b_ub - problem.A_ub.dot(lb_finite)
    b_eq_s = problem.b_eq - problem.A_eq.dot(lb_finite)
    c_s = problem.c.copy()

    n_slacks = m_ub
    n_full = n_orig + n_slacks
    m_total = m_ub + m_eq

    if m_ub > 0:
        A_ub_s = problem.A_ub
        slk = sp.eye(m_ub, format="csr")
        row_ub = sp.hstack([A_ub_s, slk], format="csr")
    else:
        row_ub = sp.csr_matrix((0, n_full))

    if m_eq > 0:
        A_eq_s = problem.A_eq
        z = sp.csr_matrix((m_eq, n_slacks))
        row_eq = sp.hstack([A_eq_s, z], format="csr")
    else:
        row_eq = sp.csr_matrix((0, n_full))

    A_full = sp.vstack([row_ub, row_eq], format="csr")
    b_full = np.concatenate([b_ub_s, b_eq_s])

    # Flip rows with negative RHS
    neg = b_full < -FEASIBILITY_TOL
    if neg.any():
        A_lil = A_full.tolil()
        A_lil[neg, :] *= -1
        A_full = A_lil.tocsr()
        b_full[neg] *= -1

    c_full = np.concatenate([c_s, np.zeros(n_slacks)])
    return c_full, A_full, b_full, lb_finite, n_orig


# ── IPM numerics ──────────────────────────────────────────────────────────────

def _step_to_bound(v: np.ndarray, dv: np.ndarray, frac: float = 0.99) -> float:
    """Maximum step α ∈ (0,1] such that v + α dv ≥ 0, times frac."""
    mask = dv < 0.0
    if not mask.any():
        return 1.0
    return float(min(1.0, frac * (-v[mask] / dv[mask]).min()))


def _normal_equations(
    A: sp.csr_matrix,
    d: np.ndarray,          # D = X S⁻¹, shape (n,)
    rhs: np.ndarray,        # (m,)
) -> np.ndarray:
    """Solve (A D Aᵀ + ε I) Δy = rhs.  Returns Δy."""
    m = A.shape[0]
    M = A.dot(sp.diags(d, format="csr").dot(A.T)).tocsc()
    # Regularize: diagonal perturbation to prevent exact singularity
    M = M + 1e-7 * sp.eye(m, format="csc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dy = spla.spsolve(M, rhs)
    if not np.all(np.isfinite(dy)):
        # Fall back to a heavier regularization
        M2 = M + 1e-3 * sp.eye(m, format="csc")
        dy = spla.spsolve(M2, rhs)
    if not np.all(np.isfinite(dy)):
        # Last resort: dense least-squares
        dy = np.linalg.lstsq(M.toarray(), rhs, rcond=None)[0]
    return dy


def _ipm_direction(
    A: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    r_c: np.ndarray,        # complementarity target vector
) -> tuple:
    """
    Compute Newton direction (dx, dy, ds) for given r_c.

    Normal equations:  (A D Aᵀ) Δy = rhs
    where:
      D   = x / s                         (scaling, clipped)
      rhs = (b - Ax) + A D (c - Aᵀy - s) + A (r_c / s)
          = r_p       + A D r_d           + A (r_c / s)
    Then:
      Δs = (c - Aᵀy - s) - Aᵀ Δy  = r_d - Aᵀ Δy
      Δx = D (r_d - Aᵀ Δy) + r_c / s
         = D Δs + r_c / s
    """
    eps = 1e-100
    r_p = b - A.dot(x)
    r_d = c - A.T.dot(y) - s
    d = np.clip(x / (s + eps), 1e-12, 1e12)   # D = X S⁻¹

    rhs = r_p + A.dot(d * r_d) - A.dot(r_c / (s + eps))
    dy = _normal_equations(A, d, rhs)

    ds = r_d - A.T.dot(dy)
    dx = r_c / (s + eps) - d * ds

    return dx, dy, ds


# ── Starting point (Mehrotra §4 heuristic) ────────────────────────────────────

def _starting_point(
    A: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple:
    """
    Compute Mehrotra's starting point.  Returns (x0, y0, s0).
    Solves least-norm primal and dual problems.
    """
    m, n = A.shape
    eps = 1e-7

    # Least-norm x̃ = Aᵀ (AAᵀ)⁻¹ b
    AAT = (A.dot(A.T)).tocsc() + eps * sp.eye(m, format="csc")
    try:
        u = spla.spsolve(AAT, b)
    except Exception:
        return np.ones(n), np.zeros(m), np.ones(n)
    x_hat = A.T.dot(u)

    # Least-norm dual: ỹ = (AAᵀ)⁻¹ A c,  s̃ = c - Aᵀ ỹ
    try:
        y_hat = spla.spsolve(AAT, A.dot(c))
    except Exception:
        return np.ones(n), np.zeros(m), np.ones(n)
    s_hat = c - A.T.dot(y_hat)

    # Shift to strict positivity (Mehrotra 1992, p.591)
    dx = max(0.0, -1.5 * x_hat.min())
    ds = max(0.0, -1.5 * s_hat.min())
    x0 = x_hat + dx
    s0 = s_hat + ds

    xTs = float(x0 @ s0)
    dx2 = 0.5 * xTs / max(eps, s0.sum())
    ds2 = 0.5 * xTs / max(eps, x0.sum())
    x0 = x0 + dx2
    s0 = s0 + ds2

    x0 = np.maximum(x0, 1e-4)
    s0 = np.maximum(s0, 1e-4)
    return x0, y_hat, s0


# ── Main solver ───────────────────────────────────────────────────────────────

def solve_lp_ipm(problem: Problem) -> SolveResult:
    """
    Solve an LP with Mehrotra's predictor-corrector interior-point method.
    """
    c, A, b, lb_orig, n_orig = _to_standard_form_ipm(problem)
    m, n = A.shape

    if m == 0 or n == 0:
        return SolveResult("optimal", np.zeros(problem.n_vars), 0.0, 0)

    # Starting point
    x, y, s = _starting_point(A, b, c)

    gaps = []
    status = "iteration_limit"

    for iters in range(MAX_IPM_ITERS):
        mu = float(x @ s) / n

        # Residuals
        r_p = b - A.dot(x)
        r_d = c - A.T.dot(y) - s

        # Convergence check
        norm_b = max(1.0, np.linalg.norm(b))
        norm_c = max(1.0, np.linalg.norm(c))
        p_res = np.linalg.norm(r_p) / norm_b
        d_res = np.linalg.norm(r_d) / norm_c
        gap   = abs(mu) / (1.0 + abs(float(c @ x)))
        gaps.append(mu)

        if max(p_res, d_res, gap) < OPTIMALITY_TOL:
            status = "optimal"
            break

        # ── Predictor (affine-scaling, σ=0) ──────────────────────────────────
        r_c_aff = -x * s
        dx_aff, dy_aff, ds_aff = _ipm_direction(A, b, c, x, y, s, r_c_aff)

        if not np.all(np.isfinite(dx_aff)):
            # Numerical breakdown — stop early
            status = "iteration_limit"
            break

        alpha_p_aff = _step_to_bound(x, dx_aff)
        alpha_d_aff = _step_to_bound(s, ds_aff)

        # ── Centering σ = (μ_aff / μ)³ ───────────────────────────────────────
        x_aff = x + alpha_p_aff * dx_aff
        s_aff = s + alpha_d_aff * ds_aff
        mu_aff = max(1e-16, float(np.maximum(x_aff, 0) @ np.maximum(s_aff, 0))) / n
        sigma = min(1.0, (mu_aff / (mu + 1e-16)) ** 3)

        # ── Corrector step ────────────────────────────────────────────────────
        r_c_cor = -x * s + sigma * mu - dx_aff * ds_aff
        dx, dy, ds = _ipm_direction(A, b, c, x, y, s, r_c_cor)

        if not np.all(np.isfinite(dx)):
            status = "iteration_limit"
            break

        # ── Step lengths ──────────────────────────────────────────────────────
        alpha_p = _step_to_bound(x, dx)
        alpha_d = _step_to_bound(s, ds)

        # ── Update ────────────────────────────────────────────────────────────
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds
        # Safeguard: clip to strict positivity
        x = np.maximum(x, 1e-14)
        s = np.maximum(s, 1e-14)

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

    return SolveResult(status, x_orig, obj_val, iters, gaps=gaps)
