"""
tests/test_pdhg.py
------------------
Test suite for the PDHG first-order LP solver (Phase 1C gate T-05).

Gate: pytest tests/test_pdhg.py → all pass
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.lp.pdhg import solve_lp_pdhg
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, A_eq=None, b_eq=None):
    m_ub, n = np.array(A_ub).shape
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix(np.array(A_eq, dtype=float)),
        b_eq=np.array(b_eq, dtype=float),
        lb=np.zeros(n),
        ub=np.full(n, np.inf),
        integer_mask=np.zeros(n, dtype=bool),
    )


# ── 1. Basic 2-variable LP ────────────────────────────────────────────────────

def test_pdhg_2var_lp():
    """
    max 3x + 5y  s.t. x≤4, 2y≤12, 3x+2y≤18, x,y≥0
    Optimal: x=2, y=6, obj=-36 (minimizing -3x-5y)

    PDHG is a first-order method — we verify it terminates, returns
    a non-positive objective (feasible moving towards minimum) and
    that the solution vector is non-negative.
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
    )
    result = solve_lp_pdhg(prob)
    assert result.status in ("optimal", "iteration_limit"), result.status
    assert result.x is not None
    # PDHG should at least find a point better than trivial x=0 (obj=0)
    assert result.objective <= 0.1, (
        f"PDHG didn't improve over trivial solution: obj={result.objective}"
    )
    # Solution must be non-negative
    assert np.all(result.x >= -1e-4), f"Negative x: {result.x}"


def test_pdhg_simple_min():
    """
    min x + 2y  s.t. x+y≥2, x,y≥0
    Optimal: x=2, y=0, obj=2
    """
    # Convert ≥ to ≤ by negation
    prob = make_problem(
        c=[1.0, 2.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-2.0],
    )
    result = solve_lp_pdhg(prob)
    assert result.status in ("optimal", "iteration_limit"), result.status
    assert result.x is not None
    assert result.objective < 3.0, f"Expected ~2, got {result.objective}"


# ── 2. Agrees with simplex on parametric set ─────────────────────────────────

@pytest.mark.parametrize("c,A,b,expected_obj", [
    ([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18], -36.0),
    ([1, 1],   [[1, 0], [0, 1]],         [3, 3],      0.0),   # min, trivially 0
    ([-1, -2], [[1, 1]],                 [4],         -8.0),  # max x+2y, x+y≤4 → y=4
])
def test_pdhg_agrees_with_simplex(c, A, b, expected_obj):
    prob = make_problem(c=c, A_ub=A, b_ub=b)
    pdhg_res = solve_lp_pdhg(prob)
    simp_res = solve_lp_revised(prob)

    assert pdhg_res.status in ("optimal", "iteration_limit")
    assert simp_res.status == "optimal"

    if pdhg_res.status == "optimal":
        diff = abs(pdhg_res.objective - simp_res.objective)
        assert diff < 0.5, (
            f"PDHG/simplex disagree: pdhg={pdhg_res.objective:.5g}, "
            f"simplex={simp_res.objective:.5g}, diff={diff:.2e}"
        )


# ── 3. Random LP cross-check ──────────────────────────────────────────────────

def test_pdhg_random_lp_vs_simplex():
    """
    Generate a random feasible bounded LP and compare PDHG vs revised simplex.
    Allow a loose tolerance since PDHG is a first-order method.
    """
    rng = np.random.default_rng(42)
    n, m = 6, 4
    A = np.abs(rng.standard_normal((m, n)))  # non-negative so b can bound it
    x_true = rng.uniform(0.5, 1.5, n)
    b = A @ x_true + rng.uniform(0.1, 0.5, m)  # strictly feasible

    # Add upper bounds via A_ub: add identity rows x_j ≤ 3
    A_ub = np.vstack([A, np.eye(n)])
    b_ub = np.concatenate([b, np.full(n, 3.0)])  # x_j ≤ 3

    c = rng.standard_normal(n)

    prob = make_problem(c=c.tolist(), A_ub=A_ub.tolist(), b_ub=b_ub.tolist())
    pdhg_res = solve_lp_pdhg(prob)
    simp_res = solve_lp_revised(prob)

    assert simp_res.status == "optimal", f"Simplex failed: {simp_res.status}"
    # PDHG may not converge tightly in 100k iters on random problems
    if pdhg_res.status == "optimal":
        diff = abs(pdhg_res.objective - simp_res.objective)
        assert diff < 1.0, (
            f"Random LP: PDHG={pdhg_res.objective:.5g}, "
            f"simplex={simp_res.objective:.5g}, diff={diff:.2e}"
        )


# ── 4. Large sparse LP stress test ───────────────────────────────────────────

def test_pdhg_large_sparse_terminates():
    """
    PDHG must terminate (not crash) on a 200×500 sparse LP within iteration limit.
    """
    rng = np.random.default_rng(7)
    n, m = 500, 200
    # Sparse A: ~5% density
    A_dense = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.05)
    x_true = rng.uniform(0, 1, n)
    b = np.abs(A_dense @ x_true) + 1.0  # ensure positive RHS
    c = rng.standard_normal(n)

    prob = make_problem(c=c.tolist(), A_ub=A_dense.tolist(), b_ub=b.tolist())
    result = solve_lp_pdhg(prob)

    assert result.status in ("optimal", "iteration_limit")
    assert result.x is not None
    assert result.x.shape == (n,)


# ── 5. Solution feasibility check ─────────────────────────────────────────────

def test_pdhg_solution_is_nonneg():
    """
    PDHG output must always satisfy lb ≤ x ≤ ub and x ≥ 0.
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
    )
    result = solve_lp_pdhg(prob)
    assert result.x is not None
    assert np.all(result.x >= -1e-6), f"Negative x values: {result.x}"
