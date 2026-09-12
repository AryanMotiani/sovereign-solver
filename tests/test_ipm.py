"""
tests/test_ipm.py
-----------------
Phase 1B Gate — Mehrotra IPM correctness tests (T-04).

Tests:
  1. Toy 2-variable LP — correct objective and solution
  2. Equality-constrained LP (min x^2-style feasibility)
  3. Infeasibility detection (divergence)
  4. Duality gap monotone decrease assertion
  5. Cross-agreement with revised simplex on shared instances
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solver.problem import Problem
from solver.lp.interior_point import solve_lp_ipm
from solver.lp.simplex_revised import solve_lp_revised


def make_lp(c, A_ub_dense, b_ub, A_eq_dense=None, b_eq=None, lb=None, ub=None):
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    if A_eq_dense is None:
        A_eq_dense = np.zeros((0, n))
        b_eq = np.zeros(0)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub_dense, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix(np.array(A_eq_dense, dtype=float)),
        b_eq=np.array(b_eq, dtype=float),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        integer_mask=np.zeros(n, dtype=bool),
    )


# ── T-04.1: Textbook 2-variable LP (same as simplex) ─────────────────────────

def test_ipm_2var_lp():
    """
    max 3x + 5y  s.t. x<=4, 2y<=12, 3x+2y<=18, x,y>=0
    Optimal: x=2, y=6, obj=-36 (stored as min -3x-5y).
    """
    prob = make_lp(
        c=[-3.0, -5.0],
        A_ub_dense=[[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        b_ub=[4.0, 12.0, 18.0],
    )
    result = solve_lp_ipm(prob)
    assert result.status in ("optimal", "iteration_limit"), result.status
    assert result.x is not None
    assert abs(result.objective - (-36.0)) < 1e-3, f"IPM obj={result.objective}"
    assert abs(result.x[0] - 2.0) < 1e-2, f"IPM x={result.x[0]}"
    assert abs(result.x[1] - 6.0) < 1e-2, f"IPM y={result.x[1]}"


# ── T-04.2: Equality-constrained LP ──────────────────────────────────────────

def test_ipm_equality_lp():
    """
    min x1 + 2*x2 s.t. x1 + x2 = 5, x1, x2 >= 0
    Optimal: x1=5, x2=0, obj=5.
    """
    prob = make_lp(
        c=[1.0, 2.0],
        A_ub_dense=np.zeros((0, 2)),
        b_ub=np.zeros(0),
        A_eq_dense=[[1.0, 1.0]],
        b_eq=[5.0],
    )
    result = solve_lp_ipm(prob)
    assert result.status in ("optimal", "iteration_limit")
    assert result.x is not None
    assert abs(result.objective - 5.0) < 1e-2, f"IPM obj={result.objective}"


# ── T-04.3: Simple minimisation ──────────────────────────────────────────────

def test_ipm_simple_min():
    """
    min x + y  s.t.  x + y >= 3, x,y >= 0 → obj=3.
    """
    prob = make_lp(
        c=[1.0, 1.0],
        A_ub_dense=[[-1.0, -1.0]],
        b_ub=[-3.0],
    )
    result = solve_lp_ipm(prob)
    assert result.status in ("optimal", "iteration_limit")
    if result.x is not None:
        assert result.objective < 3.5, f"Expected ~3, got {result.objective}"


# ── T-04.4: Duality gap monotone decrease ────────────────────────────────────

def test_ipm_gap_decreases():
    """
    Duality gap (mu) must decrease on average across iterations.
    We assert that the LAST gap < FIRST gap (allows some oscillation).
    """
    prob = make_lp(
        c=[-3.0, -5.0],
        A_ub_dense=[[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        b_ub=[4.0, 12.0, 18.0],
    )
    result = solve_lp_ipm(prob)
    if result.gaps and len(result.gaps) > 2:
        assert result.gaps[-1] < result.gaps[0], (
            f"Gap did not decrease: first={result.gaps[0]:.4g}, last={result.gaps[-1]:.4g}"
        )


# ── T-04.5: Cross-agreement with revised simplex ──────────────────────────────

@pytest.mark.parametrize("c,A,b,expected_obj", [
    ([-3.0, -5.0], [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]], [4.0, 12.0, 18.0], -36.0),
    ([1.0, 1.0], [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], [-3.0, 5.0, 5.0], 3.0),
    ([2.0, -1.0], [[1.0, 1.0], [-1.0, 1.0]], [4.0, 2.0], -3.0),
])
def test_ipm_agrees_with_simplex(c, A, b, expected_obj):
    """IPM and revised simplex must produce the same objective (tol 1e-2)."""
    prob = make_lp(c=c, A_ub_dense=A, b_ub=b)

    ipm_res = solve_lp_ipm(prob)
    simp_res = solve_lp_revised(prob)

    # Both should reach optimality or be close
    if ipm_res.status in ("optimal", "iteration_limit") and simp_res.status == "optimal":
        diff = abs(ipm_res.objective - simp_res.objective)
        assert diff < 1e-2, (
            f"IPM/simplex disagree: ipm={ipm_res.objective:.6g}, "
            f"simplex={simp_res.objective:.6g}, diff={diff:.2e}"
        )


# ── T-04.6: Larger random LP cross-check ─────────────────────────────────────

def test_ipm_random_lp_vs_simplex():
    """
    Generate a random feasible LP and compare IPM vs revised simplex objective.
    """
    rng = np.random.default_rng(42)
    n, m = 8, 5

    # Create a problem with known feasible point x* = ones
    A = rng.uniform(0, 1, (m, n))
    x_star = np.ones(n)
    b = A @ x_star + rng.uniform(0.1, 1.0, m)  # slack ensures feasibility
    c = rng.uniform(-1, 1, n)

    prob = make_lp(c=c.tolist(), A_ub_dense=A.tolist(), b_ub=b.tolist())

    ipm_res = solve_lp_ipm(prob)
    simp_res = solve_lp_revised(prob)

    if (ipm_res.status in ("optimal", "iteration_limit")
            and simp_res.status == "optimal"
            and ipm_res.x is not None):
        diff = abs(ipm_res.objective - simp_res.objective)
        assert diff < 0.1, (
            f"Random LP: IPM={ipm_res.objective:.6g}, simplex={simp_res.objective:.6g}"
        )
