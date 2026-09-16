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

Hardening additions:
  - Ruiz scaling integration for conditioning.
  - Iterative refinement for normal equations.
  - Dynamic regularization proportional to μ.
  - Full upper-bound handling in standard form.
  - Simplex crossover to basic feasible solution.

References:
    Wright (1997) "Primal-Dual Interior-Point Methods", SIAM, §6.3.
    Mehrotra (1992) SIAM J. Optim. 2(4):575-601.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import (
    FEASIBILITY_TOL,
    IPM_BARRIER_COEFF,
    IPM_ITERATIVE_REFINEMENT_STEPS,
    IPM_REGULARIZE_INIT,
    MAX_IPM_ITERS,
    OPTIMALITY_TOL,
    SCALING_METHOD,
)
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem
from solver.utils.scaling import scale_problem, unscale_solution


# ── Standard-form conversion ──────────────────────────────────────────────────

def _to_standard_form_ipm(problem: Problem) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Convert LP to equality standard form:  min cᵀx  s.t. Ax=b, x≥0.
    Handles general inequalities, variable upper bounds, and equality constraints.
    Returns (c_full, A_full, b_full, lb_finite, n_orig).
    """
    n_orig = problem.n_vars
    m_ub = problem.n_ineq
    m_eq = problem.n_eq

    lb = problem.lb.copy()
    ub = problem.ub.copy()
    lb_finite = np.where(np.isfinite(lb), lb, 0.0)

    b_ub_s = problem.b_ub - problem.A_ub.dot(lb_finite) if m_ub > 0 else np.zeros(0)
    b_eq_s = problem.b_eq - problem.A_eq.dot(lb_finite) if m_eq > 0 else np.zeros(0)
    c_s = problem.c.copy()

    # Collect finite upper-bound rows: x_j' + s_j = ub_j - lb_finite_j
    finite_ub_mask = np.isfinite(ub)
    finite_ub_cols = np.where(finite_ub_mask)[0]
    n_ub_rows = len(finite_ub_cols)

    total_slacks = m_ub + n_ub_rows
    n_full = n_orig + total_slacks
    m_total = m_ub + n_ub_rows + m_eq

    if m_total == 0:
        return c_s, sp.csr_matrix((0, n_full)), np.zeros(0), lb_finite, n_orig

    rows = []
    b_parts = []

    # 1. Original inequalities: A_ub x' + s = b_ub_s
    if m_ub > 0:
        A_ub_ext = sp.hstack([
            problem.A_ub,
            sp.eye(m_ub, format="csr"),
            sp.csr_matrix((m_ub, n_ub_rows)),
        ], format="csr")
        rows.append(A_ub_ext)
        b_parts.append(b_ub_s)

    # 2. Variable upper bounds: x'_j + s_ub = ub_j - lb_j
    if n_ub_rows > 0:
        ub_shift = ub[finite_ub_cols] - lb_finite[finite_ub_cols]
        A_ub_vars = sp.lil_matrix((n_ub_rows, n_full), dtype=float)
        for i, col in enumerate(finite_ub_cols):
            A_ub_vars[i, col] = 1.0
            A_ub_vars[i, n_orig + m_ub + i] = 1.0
        rows.append(A_ub_vars.tocsr())
        b_parts.append(ub_shift)

    # 3. Equalities: A_eq x' = b_eq_s
    if m_eq > 0:
        A_eq_ext = sp.hstack([
            problem.A_eq,
            sp.csr_matrix((m_eq, total_slacks)),
        ], format="csr")
        rows.append(A_eq_ext)
        b_parts.append(b_eq_s)

    A_full = sp.vstack(rows, format="csr")
    b_full = np.concatenate(b_parts)

    # Flip rows with negative RHS
    neg = b_full < -FEASIBILITY_TOL
    if neg.any():
        A_lil = A_full.tolil()
        A_lil[neg, :] *= -1
        A_full = A_lil.tocsr()
        b_full[neg] *= -1

    c_full = np.concatenate([c_s, np.zeros(total_slacks)])
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
    mu: float = 1.0,
) -> np.ndarray:
    """Solve (A D Aᵀ + reg I) Δy = rhs with iterative refinement."""
    m = A.shape[0]
    M = A.dot(sp.diags(d, format="csr").dot(A.T)).tocsc()

    # Dynamic regularization proportional to μ
    reg = max(1e-12, min(IPM_REGULARIZE_INIT, 1e-4 * mu))
    M_reg = M + reg * sp.eye(m, format="csc")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dy = spla.spsolve(M_reg, rhs)
        except Exception:
            dy = np.linalg.lstsq(M_reg.toarray(), rhs, rcond=None)[0]

    # Iterative refinement steps
    for _ in range(IPM_ITERATIVE_REFINEMENT_STEPS):
        res = rhs - M_reg.dot(dy)
        if np.linalg.norm(res) < 1e-11:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                corr = spla.spsolve(M_reg, res)
                dy = dy + corr
            except Exception:
                break

    if not np.all(np.isfinite(dy)):
        M_heavy = M + 1e-3 * sp.eye(m, format="csc")
        dy = np.linalg.lstsq(M_heavy.toarray(), rhs, rcond=None)[0]

    return dy


def _ipm_direction(
    A: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    r_c: np.ndarray,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Newton direction (dx, dy, ds) for given complementarity target r_c."""
    eps = 1e-100
    r_p = b - A.dot(x)
    r_d = c - A.T.dot(y) - s
    d = np.clip(x / (s + eps), 1e-12, 1e12)

    rhs = r_p + A.dot(d * r_d) - A.dot(r_c / (s + eps))
    dy = _normal_equations(A, d, rhs, mu=mu)

    ds = r_d - A.T.dot(dy)
    dx = r_c / (s + eps) - d * ds

    return dx, dy, ds


# ── Starting point (Mehrotra §4 heuristic) ────────────────────────────────────

def _starting_point(
    A: sp.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mehrotra's starting point: least-norm (x0, y0, s0)."""
    m, n = A.shape
    eps = 1e-7

    AAT = (A.dot(A.T)).tocsc() + eps * sp.eye(m, format="csc")
    try:
        u = spla.spsolve(AAT, b)
    except Exception:
        return np.ones(n), np.zeros(m), np.ones(n)
    x_hat = A.T.dot(u)

    try:
        y_hat = spla.spsolve(AAT, A.dot(c))
    except Exception:
        return np.ones(n), np.zeros(m), np.ones(n)
    s_hat = c - A.T.dot(y_hat)

    dx = max(0.0, -1.5 * float(x_hat.min()))
    ds = max(0.0, -1.5 * float(s_hat.min()))
    x0 = x_hat + dx
    s0 = s_hat + ds

    xTs = float(x0 @ s0)
    dx2 = 0.5 * xTs / max(eps, float(s0.sum()))
    ds2 = 0.5 * xTs / max(eps, float(x0.sum()))
    x0 = x0 + dx2
    s0 = s0 + ds2

    x0 = np.maximum(x0, 1e-4)
    s0 = np.maximum(s0, 1e-4)
    return x0, y_hat, s0


# ── Main solver ───────────────────────────────────────────────────────────────

def solve_lp_ipm(
    problem: Problem,
    crossover: bool = False,
    use_scaling: bool = False,
) -> SolveResult:
    """
    Solve an LP with Mehrotra's predictor-corrector interior-point method.

    Parameters
    ----------
    problem : Problem
    crossover : bool
        If True, run simplex crossover to find a basic feasible solution.
    use_scaling : bool
        If True, apply Ruiz scaling prior to IPM solve.

    Returns
    -------
    SolveResult
    """
    work_problem = problem
    scale_factors = None

    if use_scaling and SCALING_METHOD in ("ruiz", "geometric"):
        work_problem, scale_factors = scale_problem(problem, method=SCALING_METHOD)

    c, A, b, lb_orig, n_orig = _to_standard_form_ipm(work_problem)
    m, n = A.shape

    if m == 0 or n == 0:
        x_orig = np.clip(
            np.zeros(problem.n_vars) if lb_orig is None else lb_orig,
            np.where(np.isfinite(problem.lb), problem.lb, -1e30),
            np.where(np.isfinite(problem.ub), problem.ub, 1e30),
        )
        obj_val = float(problem.c @ x_orig)
        if problem.sense == "max":
            obj_val = -obj_val
        return SolveResult("optimal", x_orig, obj_val, 0)

    # Starting point
    x, y, s = _starting_point(A, b, c)

    gaps = []
    status = "iteration_limit"
    iters = 0

    for iters in range(1, MAX_IPM_ITERS + 1):
        mu = float(x @ s) / n

        # Residuals
        r_p = b - A.dot(x)
        r_d = c - A.T.dot(y) - s

        # Convergence check
        norm_b = max(1.0, float(np.linalg.norm(b)))
        norm_c = max(1.0, float(np.linalg.norm(c)))
        p_res = float(np.linalg.norm(r_p)) / norm_b
        d_res = float(np.linalg.norm(r_d)) / norm_c
        gap   = abs(mu) / (1.0 + abs(float(c @ x)))
        gaps.append(mu)

        if max(p_res, d_res, gap) < OPTIMALITY_TOL:
            status = "optimal"
            break

        # ── Predictor (affine-scaling, σ=0) ──────────────────────────────────
        r_c_aff = -x * s
        dx_aff, dy_aff, ds_aff = _ipm_direction(A, b, c, x, y, s, r_c_aff, mu=mu)

        if not np.all(np.isfinite(dx_aff)):
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
        dx, dy, ds = _ipm_direction(A, b, c, x, y, s, r_c_cor, mu=mu)

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
        x = np.maximum(x, 1e-14)
        s = np.maximum(s, 1e-14)

    # ── Extract solution ──────────────────────────────────────────────────────
    x_shifted = x[:n_orig]
    x_orig = x_shifted + lb_orig

    if scale_factors is not None:
        x_orig = unscale_solution(x_orig, scale_factors)

    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    # ── Crossover to basic solution if requested ──────────────────────────────
    if crossover and status == "optimal":
        from solver.lp.simplex_revised import solve_lp_revised
        cross_res = solve_lp_revised(problem)
        if cross_res.status == "optimal":
            return SolveResult(
                "optimal", cross_res.x, cross_res.objective, iters + cross_res.iterations,
                basis=cross_res.basis, gaps=gaps
            )

    return SolveResult(status, x_orig, obj_val, iters, gaps=gaps)
