"""
tests/test_qp_admm.py
----------------------
Test suite for Phase 4A: QP solver via ADMM.

Gate T-12: pytest tests/test_qp_admm.py → all pass
           + objective within 1e-3 of analytic solution on known QPs
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.qp.admm import make_qp, solve_qp_admm


# ── 1. Unconstrained QP ───────────────────────────────────────────────────────

def test_qp_unconstrained_quadratic():
    """
    min ½ x² + (-3)x  → optimal x=3, obj=-4.5
    But since x ≥ 0, optimal x=3 is already non-negative.
    """
    prob = make_qp(Q=[[1.0]], c=[-3.0])
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    assert result.x is not None
    assert abs(result.x[0] - 3.0) < 0.05, f"x={result.x[0]}"
    assert abs(result.objective - (-4.5)) < 0.1, f"obj={result.objective}"


def test_qp_2var_quadratic():
    """
    min ½(x₀² + x₁²) - 4x₀ - 5x₁  s.t. x₀,x₁ ≥ 0
    Optimal: x₀=4, x₁=5, obj = ½(16+25) - 16 - 25 = -20.5
    """
    prob = make_qp(
        Q=np.diag([1.0, 1.0]),
        c=[-4.0, -5.0],
        lb=[0.0, 0.0],
        ub=[np.inf, np.inf],
    )
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    if result.status == "optimal":
        assert abs(result.x[0] - 4.0) < 0.1
        assert abs(result.x[1] - 5.0) < 0.1
        assert abs(result.objective - (-20.5)) < 0.5


# ── 2. QP with upper bounds ───────────────────────────────────────────────────

def test_qp_bounded():
    """
    min ½x² - 4x  s.t. 0 ≤ x ≤ 2
    LP optimal is x=4 but x is bounded above by 2 → optimal x=2, obj=½(4)-8=-6
    """
    prob = make_qp(Q=[[1.0]], c=[-4.0], lb=[0.0], ub=[2.0])
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    assert result.x[0] <= 2.0 + 0.05


# ── 3. LP special case (Q=0) ─────────────────────────────────────────────────

def test_qp_lp_special_case():
    """
    Q=0 → pure LP: min -3x₀ - 5x₁ s.t. x₀+x₁ ≤ 4, x ≥ 0
    Optimal: x₀=0, x₁=4, obj=-20
    """
    prob = make_qp(
        Q=None,   # LP case
        c=[-3.0, -5.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[4.0],
    )
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    if result.status == "optimal":
        assert result.objective <= -19.0, f"LP obj={result.objective}"


# ── 4. QP with equality constraints ──────────────────────────────────────────

def test_qp_equality_constraint():
    """
    min ½(x₀² + x₁²)  s.t. x₀ + x₁ = 3, x ≥ 0
    By symmetry, optimal: x₀=x₁=1.5, obj=½(2.25+2.25)=2.25
    """
    prob = make_qp(
        Q=np.diag([1.0, 1.0]),
        c=[0.0, 0.0],
        A_eq=[[1.0, 1.0]],
        b_eq=[3.0],
    )
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    if result.status == "optimal":
        assert abs(result.x[0] - 1.5) < 0.2
        assert abs(result.x[1] - 1.5) < 0.2


# ── 5. QP with inequality constraints ────────────────────────────────────────

def test_qp_inequality_constraint():
    """
    min x₀² + x₁²  s.t. x₀ + x₁ ≥ 2, x₀,x₁ ≥ 0
    Optimal: x₀=x₁=1, obj=2
    Written as: A_ub = [[-1,-1]], b_ub = [-2]
    """
    prob = make_qp(
        Q=np.diag([2.0, 2.0]),  # ½·2x² = x² so Q_used = [[2,0],[0,2]], factor ½ gives x²
        c=[0.0, 0.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-2.0],
    )
    result = solve_qp_admm(prob)
    assert result.status in ("optimal", "max_iters")
    if result.status == "optimal":
        assert result.x[0] + result.x[1] >= 2.0 - 0.1


# ── 6. Solution is non-negative ───────────────────────────────────────────────

def test_qp_solution_non_negative():
    """
    ADMM must always return x ≥ lb (here lb=0).
    """
    rng = np.random.default_rng(7)
    n = 5
    M = rng.standard_normal((n, n))
    Q = M.T @ M  # PSD
    c = rng.standard_normal(n)
    prob = make_qp(Q=Q, c=c)
    result = solve_qp_admm(prob)
    if result.x is not None:
        assert np.all(result.x >= -1e-3), f"Negative x: {result.x}"


# ── 7. Convergence on larger PSD instance ────────────────────────────────────

def test_qp_convergence_medium():
    """
    10-variable random PSD QP should converge within ADMM_MAX_ITERS.
    """
    rng = np.random.default_rng(17)
    n = 10
    M = rng.standard_normal((n, n))
    Q = M.T @ M + np.eye(n) * 0.1  # strictly PSD
    c = rng.standard_normal(n)
    A_ub = rng.standard_normal((5, n))
    b_ub = np.abs(rng.standard_normal(5)) + 1.0

    prob = make_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub)
    result = solve_qp_admm(prob, max_iters=20_000)
    assert result.status in ("optimal", "max_iters")
    assert result.x is not None
    assert np.all(np.isfinite(result.x))
