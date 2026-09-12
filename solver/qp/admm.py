"""
solver/qp/admm.py
------------------
Quadratic Program (QP) solver via ADMM (Alternating Direction Method of Multipliers).

Solves:
    min  ½ xᵀQx + cᵀx
    s.t. Ax = b,  x ≥ 0

where Q is a positive semi-definite (PSD) matrix.

ADMM formulation (Boyd et al. 2010):
  Augmented Lagrangian split: introduce auxiliary z ≥ 0, constraint x = z.

  x-update:  min ½ xᵀQx + cᵀx + (ρ/2)‖x - z + u‖²   (unconstrained QP with linear solve)
  z-update:  z = max(x + u, 0)                          (projection onto x ≥ 0)
  u-update:  u ← u + x - z                              (dual variable update)

  Termination: primal residual ‖x - z‖ and dual residual ‖ρ(z - z_old)‖ both < tol.

Extensions:
  - Linear (LP) problems: Q=0 case handled via standard ADMM with identity penalty.
  - Equality constraints: added as additional penalty ρ₂‖Ax - b‖²/2.

References:
    Boyd et al. (2010) "Distributed Optimization via ADMM", Foundations & Trends.
    Stellato et al. (2020) "OSQP: An Operator Splitting Solver for QPs", Math. Prog.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import (
    ADMM_MAX_ITERS,
    ADMM_RHO,
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
    status:    str            # 'optimal' | 'max_iters' | 'infeasible'
    x:         Optional[np.ndarray]
    objective: float
    iters:     int
    primal_res: float = 0.0
    dual_res:   float = 0.0


# ── ADMM solver ───────────────────────────────────────────────────────────────

def solve_qp_admm(
    problem: QPProblem,
    rho: float = ADMM_RHO,
    max_iters: int = ADMM_MAX_ITERS,
    tol_abs: float = ADMM_TOL_ABS,
    tol_rel: float = ADMM_TOL_REL,
    verbose: bool = False,
) -> QPResult:
    """
    Solve a QP using ADMM.

    Strategy:
      1. Convert inequality constraints Aᵤ x ≤ bᵤ to equalities by adding
         slack variables: introduce s ≥ 0 with Aᵤ x + s = bᵤ.
      2. Stack [x; s] into a single extended variable w.
      3. Run ADMM on the extended problem with x-update via a cached factorization.

    Parameters
    ----------
    problem  : QPProblem
    rho      : ADMM penalty parameter
    max_iters: max iterations
    tol_abs  : absolute residual tolerance
    tol_rel  : relative residual tolerance

    Returns
    -------
    QPResult
    """
    n    = problem.n
    m_eq = problem.m_eq
    m_ub = problem.m_ub
    n_s  = m_ub             # slack variables for inequalities
    n_w  = n + n_s          # extended variable dimension: [x; s]

    # ── Build extended system ──────────────────────────────────────────────────
    # Extended Q_ext block: Q on x-part, 0 on slack part
    Q_ext = sp.block_diag([problem.Q, sp.csr_matrix((n_s, n_s))], format="csr")
    c_ext = np.concatenate([problem.c, np.zeros(n_s)])

    # Equality constraints on extended variable:
    #   [A_eq | 0  ] w = b_eq
    #   [A_ub | I  ] w = b_ub   (inequality converted via slack)
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

    # Bound: lb_w = [lb | 0], ub_w = [ub | inf]
    lb_w = np.concatenate([problem.lb, np.zeros(n_s)])
    ub_w = np.concatenate([problem.ub, np.full(n_s, np.inf)])

    # ── KKT factorization for x-update ────────────────────────────────────────
    # Saddle-point system:
    #   [Q_ext + ρI   A_ext^T] [w ] = [-c_ext + ρ(z-u)]
    #   [A_ext        0      ] [ν ] = [b_ext            ]
    # Factor once, solve each iteration for right-hand side.
    n_kkt = n_w + m_ext
    KKT_top = sp.hstack([Q_ext + rho * sp.eye(n_w, format="csr"), A_ext.T], format="csr")
    if m_ext > 0:
        KKT_bot = sp.hstack([A_ext, sp.csr_matrix((m_ext, m_ext))], format="csr")
        KKT = sp.vstack([KKT_top, KKT_bot], format="csc")
    else:
        KKT = KKT_top.tocsc()

    try:
        factor = spla.splu(KKT)
    except Exception:
        return QPResult("infeasible", None, np.inf, 0)

    # ── Initialize ─────────────────────────────────────────────────────────────
    w = np.zeros(n_w)
    z = np.zeros(n_w)    # auxiliary (projection target)
    u = np.zeros(n_w)    # scaled dual variable

    status = "max_iters"
    primal_res = dual_res = np.inf

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
        primal_res = np.linalg.norm(w - z)
        dual_res   = rho * np.linalg.norm(z - z_old)

        eps_prim = np.sqrt(n_w) * tol_abs + tol_rel * max(np.linalg.norm(w), np.linalg.norm(z))
        eps_dual = np.sqrt(n_w) * tol_abs + tol_rel * rho * np.linalg.norm(u)

        if primal_res < eps_prim and dual_res < eps_dual:
            status = "optimal"
            break

        if verbose and it % 500 == 0:
            print(f"  [ADMM] iter={it:6d}  prim={primal_res:.3e}  dual={dual_res:.3e}")

    x_sol = w[:n]
    obj = float(0.5 * x_sol @ (problem.Q @ x_sol) + problem.c @ x_sol)

    return QPResult(status, x_sol, obj, it, primal_res, dual_res)


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
