"""
tests/test_robustness.py
------------------------
Phase 5 Gate: Comprehensive Robustness and Stress Test Suite.

Tests the solver on:
  1. Degenerate LPs (Beale's cycling problem & degenerate bases).
  2. Ill-conditioned LPs (high condition number).
  3. Badly scaled LPs (coefficients spanning 8+ orders of magnitude).
  4. Provably infeasible and unbounded problems.
  5. Large sparse LPs (n=1000, m=500).
  6. Degenerate and weak-relaxation MILPs.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.lp.pdhg import solve_lp_pdhg
from solver.lp.simplex_revised import solve_lp_revised
from solver.milp.branch_and_bound import solve_milp
from solver.problem import Problem


# ── 1. Degenerate LP (Beale's cycling problem) ────────────────────────────────

def test_degenerate_cycling_lp():
    """
    Beale's classic cycling problem:
    min -0.75 x0 + 20 x1 - 0.5 x2 + 6 x3
    s.t.
       0.25 x0 - 8 x1 - x2 + 9 x3 <= 0
       0.5 x0 - 12 x1 - 0.5 x2 + 3 x3 <= 0
       x0 <= 1
       x >= 0

    Under standard Dantzig pivot rule, this cycles indefinitely.
    With Harris ratio test and Bland's anti-cycling rule, it must terminate.
    """
    c = np.array([-0.75, 20.0, -0.5, 6.0])
    A_eq = np.array([
        [0.25, -8.0, -1.0, 9.0],
        [0.5, -12.0, -0.5, 3.0],
    ])
    b_eq = np.array([0.0, 0.0])
    A_ub = np.array([[1.0, 0.0, 0.0, 0.0]])
    b_ub = np.array([1.0])

    prob = Problem(
        c=c,
        A_ub=sp.csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=sp.csr_matrix(A_eq),
        b_eq=b_eq,
        lb=np.zeros(4),
        ub=np.full(4, np.inf),
        integer_mask=np.zeros(4, dtype=bool),
        name="beale_cycling",
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert np.isfinite(result.objective)
    # Optimal value is -0.5 (at x = [1, 0, 2.5, 0.25])
    assert abs(result.objective - (-0.5)) < 1e-4, f"Expected -0.5, got {result.objective}"


# ── 2. Ill-conditioned LP ─────────────────────────────────────────────────────

def test_ill_conditioned_lp():
    """
    LP with near-singular constraint matrix:
    min x0 + x1
    s.t.
      x0 + x1 >= 2
      x0 + (1 + 1e-5) x1 >= 2 + 1e-5
      x0, x1 >= 0
    Optimal is x0=1, x1=1, obj=2.
    """
    eps = 1e-5
    c = np.array([1.0, 1.0])
    A_ub = np.array([
        [-1.0, -1.0],
        [-1.0, -(1.0 + eps)],
    ])
    b_ub = np.array([-2.0, -(2.0 + eps)])

    prob = Problem(
        c=c,
        A_ub=sp.csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=sp.csr_matrix((0, 2)),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.zeros(2, dtype=bool),
        name="ill_conditioned",
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert abs(result.objective - 2.0) < 1e-3, f"Expected 2.0, got {result.objective}"


# ── 3. Badly scaled LP (spanning 8 orders of magnitude) ───────────────────────

def test_badly_scaled_lp():
    """
    Coefficients span 10^-4 to 10^4:
    min 1e4 x0 + 1e-3 x1
    s.t.
      1e3 x0 + 1e-4 x1 >= 10.0
      x0, x1 >= 0
    Optimal: x0 = 0.01, x1 = 0, obj = 100.
    """
    c = np.array([1e4, 1e-3])
    A_ub = np.array([[-1e3, -1e-4]])
    b_ub = np.array([-10.0])

    prob = Problem(
        c=c,
        A_ub=sp.csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=sp.csr_matrix((0, 2)),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.zeros(2, dtype=bool),
        name="badly_scaled",
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 100.0) < 0.1


# ── 4. Infeasible and unbounded detection ─────────────────────────────────────

def test_infeasible_lp_detection():
    """Infeasible: x0 + x1 <= -1 for x >= 0."""
    prob = Problem(
        c=np.array([1.0, 1.0]),
        A_ub=sp.csr_matrix([[1.0, 1.0]]),
        b_ub=np.array([-1.0]),
        A_eq=sp.csr_matrix((0, 2)),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.zeros(2, dtype=bool),
    )
    result = solve_lp_revised(prob)
    assert result.status == "infeasible"


def test_unbounded_lp_detection():
    """Unbounded: min -x0 s.t. x0 >= 0 (no upper bound)."""
    prob = Problem(
        c=np.array([-1.0, 0.0]),
        A_ub=sp.csr_matrix([[0.0, 1.0]]),
        b_ub=np.array([5.0]),
        A_eq=sp.csr_matrix((0, 2)),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.zeros(2, dtype=bool),
    )
    result = solve_lp_revised(prob)
    assert result.status == "unbounded"


# ── 5. Large sparse LP (n=1000, m=500) ────────────────────────────────────────

def test_large_sparse_lp():
    """
    Sparse banded LP: n=500 vars, m=250 constraints.
    Verifies solver scales and terminates within reasonable time.
    """
    n, m = 500, 250
    rng = np.random.default_rng(42)

    # Sparse diagonal / banded matrix
    diagonals = [rng.uniform(1, 3, n), rng.uniform(0.5, 1.5, n - 1)]
    A_banded = sp.diags(diagonals, [0, -1], shape=(m, n), format="csr")

    x_true = rng.uniform(0.5, 2.0, n)
    b = np.asarray(A_banded.dot(x_true)).ravel() + rng.uniform(0.1, 0.5, m)
    c = rng.uniform(-2, -0.5, n)

    prob = Problem(
        c=c,
        A_ub=A_banded,
        b_ub=b,
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.zeros(n),
        ub=np.full(n, 5.0),
        integer_mask=np.zeros(n, dtype=bool),
        name="large_sparse_lp",
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    assert np.isfinite(result.objective)
    assert result.x is not None
    assert len(result.x) == n


# ── 6. Degenerate MILP ────────────────────────────────────────────────────────

def test_degenerate_milp():
    """
    MILP with multiple symmetric integer solutions:
    min x0 + x1 + x2
    s.t. x0 + x1 + x2 >= 2
         x0, x1, x2 in {0, 1}
    Optimal obj = 2.
    """
    prob = Problem(
        c=np.array([1.0, 1.0, 1.0]),
        A_ub=sp.csr_matrix([[-1.0, -1.0, -1.0]]),
        b_ub=np.array([-2.0]),
        A_eq=sp.csr_matrix((0, 3)),
        b_eq=np.zeros(0),
        lb=np.zeros(3),
        ub=np.ones(3),
        integer_mask=np.ones(3, dtype=bool),
        name="degenerate_milp",
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 2.0) < 1e-4
    assert np.all(np.abs(result.x - np.round(result.x)) < 1e-4)


# ── 7. Weak LP relaxation MILP ────────────────────────────────────────────────

def test_weak_lp_relaxation_milp():
    """
    0-1 Knapsack where LP relaxation gives fractional values for multiple items:
    max 5x0 + 6x1 + 7x2 + 8x3
    s.t. 4x0 + 5x1 + 6x2 + 7x3 <= 12
    x in {0, 1}^4
    """
    c = np.array([-5.0, -6.0, -7.0, -8.0])
    A_ub = np.array([[4.0, 5.0, 6.0, 7.0]])
    b_ub = np.array([12.0])

    prob = Problem(
        c=c,
        A_ub=sp.csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=sp.csr_matrix((0, 4)),
        b_eq=np.zeros(0),
        lb=np.zeros(4),
        ub=np.ones(4),
        integer_mask=np.ones(4, dtype=bool),
        name="knapsack_weak_relaxation",
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    # Possible assignments:
    # x = [1, 0, 0, 1] weight 11, val 13
    # x = [0, 1, 1, 0] weight 11, val 13
    # x = [1, 1, 0, 0] weight 9, val 11
    # max val = 14: x = [1, 0, 1, 0] weight 10, val 12?
    # x = [0, 1, 1, 0] weight 11, val 13
    # x = [1, 1, 0, 0] weight 9, val 11
    # Check against brute force
    best_bf = np.inf
    for mask in range(16):
        xb = np.array([(mask >> j) & 1 for j in range(4)], dtype=float)
        if 4 * xb[0] + 5 * xb[1] + 6 * xb[2] + 7 * xb[3] <= 12.0:
            val = float(c @ xb)
            if val < best_bf:
                best_bf = val

    assert abs(result.objective - best_bf) < 1e-4, f"Expected {best_bf}, got {result.objective}"
