"""
solver/qp/admm.py
------------------
Quadratic Program (QP) solver via ADMM (Alternating Direction Method of Multipliers).

Solves:
    min  ½ xᵀQx + cᵀx
    s.t. A_eq x = b_eq,
         A_ub x ≤ b_ub,
         lb ≤ x ≤ ub

where Q is a positive semi-definite (PSD) matrix.

ADMM formulation (Boyd et al. 2010, Stellato et al. OSQP 2020):
  Augmented Lagrangian split: introduce auxiliary z, constraint w = z where w = [x; s].

  x-update:  solve [Q_ext + ρI   A_extᵀ] [w] = [-c_ext + ρ(z - u)]
                   [A_ext        0     ] [ν]   [b_ext            ]
  z-update:  z = clip(w + u, lb_w, ub_w)
  u-update:  u ← u + w - z

Hardening features:
  - Adaptive ρ (Boyd et al. §3.4.1) balancing primal and dual residuals.
  - Warm-starting support (x_init, u_init).
  - Solution polishing (OSQP §5.2) solving reduced KKT on the active set.

References:
    Boyd et al. (2010) "Distributed Optimization via ADMM", Foundations & Trends.
    Stellato et al. (2020) "OSQP: An Operator Splitting Solver for QPs", Math. Prog.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import (
    ADMM_MAX_ITERS,
    ADMM_POLISH,
    ADMM_RHO,
    ADMM_RHO_ADAPT_MU,
    ADMM_TOL_ABS,
    ADMM_TOL_REL,
)


# ── QP Problem dataclass ──────────────────────────────────────────────────────

@dataclasses.dataclass
class QPProblem:
    """
    Standard form QP:
        min  ½ xᵀQx + cᵀx
        s.t. A_eq x = b_eq,
             A_ub x ≤ b_ub,
             lb ≤ x ≤ ub
    """
    Q:     sp.csr_matrix      # (n×n) PSD matrix; may be all-zeros for LP
    c:     np.ndarray         # (n,)
    A_eq:  sp.csr_matrix      # (m_eq × n)
    b_eq:  np.ndarray         # (m_eq,)
    A_ub:  sp.csr_matrix      # (m_ub × n)
    b_ub:  np.ndarray         # (m_ub,)
    lb:    np.ndarray         # (n,)
    ub:    np.ndarray         # (n,)

    @property
    def n(self): return self.c.shape[0]
    @property
    def m_eq(self): return self.b_eq.shape[0]
    @property
    def m_ub(self): return self.b_ub.shape[0]


@dataclasses.dataclass
class QPResult:
    status:      str            # 'optimal' | 'max_iters' | 'infeasible'
    x:           Optional[np.ndarray]
    objective:   float
    iters:       int
    primal_res:  float = 0.0
    dual_res:    float = 0.0
    u:           Optional[np.ndarray] = None   # Dual multipliers for warm-start


# ── ADMM solver ───────────────────────────────────────────────────────────────

def solve_qp_admm(
    problem: QPProblem,
    rho: float = ADMM_RHO,
    max_iters: int = ADMM_MAX_ITERS,
    tol_abs: float = ADMM_TOL_ABS,
    tol_rel: float = ADMM_TOL_REL,
    adaptive_rho: bool = True,
    polish: bool = ADMM_POLISH,
    x_init: Optional[np.ndarray] = None,
    u_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> QPResult:
    """
    Solve a QP using ADMM with adaptive rho and solution polishing.

    Parameters
    ----------
    problem : QPProblem
    rho : float
        ADMM penalty parameter.
    max_iters : int
        Maximum ADMM iterations.
    tol_abs, tol_rel : float
        Absolute and relative stopping tolerances.
    adaptive_rho : bool
        Adjust rho to balance primal and dual residuals.
    polish : bool
        Solve unconstrained KKT on active set to achieve high precision.
    x_init, u_init : Optional[np.ndarray]
        Warm-start primal and dual variables.

    Returns
    -------
    QPResult
    """
    n    = problem.n
    m_eq = problem.m_eq
    m_ub = problem.m_ub
    n_s  = m_ub
    n_w  = n + n_s

    # ── Build extended system ──────────────────────────────────────────────────
    Q_ext = sp.block_diag([problem.Q, sp.csr_matrix((n_s, n_s))], format="csr")
    c_ext = np.concatenate([problem.c, np.zeros(n_s)])

    if m_eq > 0 and m_ub > 0:
        A_top = sp.hstack([problem.A_eq, sp.csr_matrix((m_eq, n_s))], format="csr")
        A_bot = sp.hstack([problem.A_ub, sp.eye(n_s, format="csr")],  format="csr")
        A_ext = sp.vstack([A_top, A_bot], format="csr")
        b_ext = np.concatenate([problem.b_eq, problem.b_ub])
        m_ext = m_eq + m_ub
    elif m_eq > 0:
        A_ext = sp.hstack([problem.A_eq, sp.csr_matrix((m_eq, n_s))], format="csr")
        b_ext = problem.b_eq
        m_ext = m_eq
    elif m_ub > 0:
        A_ext = sp.hstack([problem.A_ub, sp.eye(n_s, format="csr")], format="csr")
        b_ext = problem.b_ub
        m_ext = m_ub
    else:
        A_ext = sp.csr_matrix((0, n_w))
        b_ext = np.zeros(0)
        m_ext = 0

    lb_w = np.concatenate([problem.lb, np.zeros(n_s)])
    ub_w = np.concatenate([problem.ub, np.full(n_s, np.inf)])

    # ── KKT factorization helper ──────────────────────────────────────────────
    def _factor_kkt(rho_val: float):
        KKT_top = sp.hstack([Q_ext + rho_val * sp.eye(n_w, format="csr"), A_ext.T], format="csr")
        if m_ext > 0:
            KKT_bot = sp.hstack([A_ext, sp.csr_matrix((m_ext, m_ext))], format="csr")
            KKT = sp.vstack([KKT_top, KKT_bot], format="csc")
        else:
            KKT = KKT_top.tocsc()
        return spla.splu(KKT)

    try:
        factor = _factor_kkt(rho)
    except Exception:
        return QPResult("infeasible", None, np.inf, 0)

    # ── Initialize ─────────────────────────────────────────────────────────────
    w = np.zeros(n_w)
    if x_init is not None:
        w[:min(n, len(x_init))] = x_init[:min(n, len(x_init))]
        if m_ub > 0:
            # Initialize slacks
            w[n:] = np.maximum(0.0, problem.b_ub - problem.A_ub.dot(w[:n]))

    z = np.clip(w.copy(), lb_w, ub_w)
    u = np.zeros(n_w)
    if u_init is not None:
        u[:min(n_w, len(u_init))] = u_init[:min(n_w, len(u_init))]

    status = "max_iters"
    primal_res = dual_res = np.inf
    adapt_interval = 25

    for it in range(1, max_iters + 1):
        # ── x-update: solve KKT system ─────────────────────────────────────────
        rhs_top = -c_ext + rho * (z - u)
        rhs = np.concatenate([rhs_top, b_ext]) if m_ext > 0 else rhs_top
        sol = factor.solve(rhs)
        w_new = sol[:n_w]

        # ── z-update: project onto box [lb_w, ub_w] ───────────────────────────
        z_old = z.copy()
        z = np.clip(w_new + u, lb_w, ub_w)

        # ── u-update: dual ascent ──────────────────────────────────────────────
        u = u + w_new - z

        # ── Convergence check ──────────────────────────────────────────────────
        w = w_new
        primal_res = float(np.linalg.norm(w - z))
        dual_res   = float(rho * np.linalg.norm(z - z_old))

        eps_prim = float(np.sqrt(n_w) * tol_abs + tol_rel * max(np.linalg.norm(w), np.linalg.norm(z)))
        eps_dual = float(np.sqrt(n_w) * tol_abs + tol_rel * rho * np.linalg.norm(u))

        if primal_res < eps_prim and dual_res < eps_dual:
            status = "optimal"
            break

        # ── Adaptive rho (Boyd et al. §3.4.1) ─────────────────────────────────
        if adaptive_rho and it % adapt_interval == 0:
            rho_changed = False
            if primal_res > ADMM_RHO_ADAPT_MU * dual_res and rho < 1e6:
                rho *= 2.0
                u /= 2.0
                rho_changed = True
            elif dual_res > ADMM_RHO_ADAPT_MU * primal_res and rho > 1e-6:
                rho /= 2.0
                u *= 2.0
                rho_changed = True

            if rho_changed:
                try:
                    factor = _factor_kkt(rho)
                except Exception:
                    pass

    x_sol = w[:n]

    # ── Solution polishing ────────────────────────────────────────────────────
    if polish and status == "optimal":
        try:
            # Active bounds: z_j near lower or upper bound
            act_lb = (z[:n] <= problem.lb + 1e-4) & np.isfinite(problem.lb)
            act_ub = (z[:n] >= problem.ub - 1e-4) & np.isfinite(problem.ub)
            act_vars = act_lb | act_ub
            free_vars = ~act_vars

            # Active inequalities (slack close to 0)
            act_ineq = (z[n:] <= 1e-4) if m_ub > 0 else np.zeros(0, dtype=bool)

            # Assemble active constraints
            A_active_parts = []
            b_active_parts = []
            if m_eq > 0:
                A_active_parts.append(problem.A_eq)
                b_active_parts.append(problem.b_eq)
            if m_ub > 0 and act_ineq.any():
                A_active_parts.append(problem.A_ub[act_ineq])
                b_active_parts.append(problem.b_ub[act_ineq])

            if free_vars.any():
                free_idx = np.where(free_vars)[0]
                act_idx = np.where(act_vars)[0]

                Q_ff = problem.Q[np.ix_(free_idx, free_idx)].tocsc()
                c_f = problem.c[free_idx].copy()

                if len(act_idx) > 0:
                    x_act = np.where(act_lb[act_idx], problem.lb[act_idx], problem.ub[act_idx])
                    c_f += problem.Q[np.ix_(free_idx, act_idx)].dot(x_act)
                else:
                    x_act = np.zeros(0)

                if A_active_parts:
                    A_act_mat = sp.vstack(A_active_parts, format="csr")
                    b_act_vec = np.concatenate(b_active_parts)
                    m_act = A_act_mat.shape[0]

                    A_act_f = A_act_mat[:, free_idx]
                    b_act_eff = b_act_vec.copy()
                    if len(act_idx) > 0:
                        b_act_eff -= A_act_mat[:, act_idx].dot(x_act)

                    K_top = sp.hstack([Q_ff, A_act_f.T], format="csr")
                    K_bot = sp.hstack([A_act_f, sp.csr_matrix((m_act, m_act))], format="csr")
                    K_polish = sp.vstack([K_top, K_bot], format="csc")
                    rhs_polish = np.concatenate([-c_f, b_act_eff])

                    sol_polish = spla.spsolve(K_polish, rhs_polish)
                    x_free_pol = sol_polish[:len(free_idx)]
                else:
                    K_polish = Q_ff + 1e-10 * sp.eye(len(free_idx), format="csc")
                    x_free_pol = spla.spsolve(K_polish, -c_f)

                x_polished = x_sol.copy()
                x_polished[free_idx] = x_free_pol
                if len(act_idx) > 0:
                    x_polished[act_idx] = x_act

                x_polished = np.clip(
                    x_polished,
                    np.where(np.isfinite(problem.lb), problem.lb, -1e30),
                    np.where(np.isfinite(problem.ub), problem.ub, 1e30),
                )

                # Feasibility check: only accept if inequality constraints hold
                feas = True
                if m_ub > 0:
                    ub_viol = problem.A_ub.dot(x_polished) - problem.b_ub
                    if np.any(ub_viol > 1e-4):
                        feas = False
                if m_eq > 0:
                    eq_viol = np.abs(problem.A_eq.dot(x_polished) - problem.b_eq)
                    if np.any(eq_viol > 1e-4):
                        feas = False

                if feas:
                    x_sol = x_polished
        except Exception:
            pass

    x_sol = np.clip(
        x_sol,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )
    obj = float(0.5 * x_sol @ (problem.Q @ x_sol) + problem.c @ x_sol)

    return QPResult(status, x_sol, obj, it, primal_res, dual_res, u=u[:n])


# ── Convenience constructor ───────────────────────────────────────────────────

def make_qp(
    Q, c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, lb=None, ub=None
) -> QPProblem:
    """
    Construct a QPProblem from dense/sparse inputs.
    Q may be None (LP case), in which case a zero matrix is used.
    """
    c = np.asarray(c, dtype=float)
    n = len(c)

    if Q is None:
        Q = sp.csr_matrix((n, n))
    else:
        Q = sp.csr_matrix(np.asarray(Q, dtype=float))

    if A_eq is None:
        A_eq = sp.csr_matrix((0, n))
        b_eq = np.zeros(0)
    else:
        A_eq = sp.csr_matrix(np.asarray(A_eq, dtype=float))
        b_eq = np.asarray(b_eq, dtype=float)

    if A_ub is None:
        A_ub = sp.csr_matrix((0, n))
        b_ub = np.zeros(0)
    else:
        A_ub = sp.csr_matrix(np.asarray(A_ub, dtype=float))
        b_ub = np.asarray(b_ub, dtype=float)

    if lb is None:
        lb = np.zeros(n)
    else:
        lb = np.asarray(lb, dtype=float)

    if ub is None:
        ub = np.full(n, np.inf)
    else:
        ub = np.asarray(ub, dtype=float)

    return QPProblem(Q=Q, c=c, A_eq=A_eq, b_eq=b_eq,
                     A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub)
