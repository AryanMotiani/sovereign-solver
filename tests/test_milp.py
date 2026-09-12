"""
tests/test_milp.py
------------------
Test suite for Phase 2B: Branch-and-Bound MILP solver.

Gate T-07: pytest tests/test_milp.py → all pass
           (including brute-force cross-check on small instances)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.milp.branch_and_bound import solve_milp, MILPSolveResult
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, integer_vars, A_eq=None, b_eq=None, lb=None, ub=None):
    n = len(c)
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    int_mask = np.zeros(n, dtype=bool)
    for i in integer_vars:
        int_mask[i] = True
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix(np.array(A_eq, dtype=float)),
        b_eq=np.array(b_eq, dtype=float),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        integer_mask=int_mask,
    )


# ── 1. Trivial pure IP ────────────────────────────────────────────────────────

def test_milp_trivial_binary():
    """
    max x0 + x1  s.t. x0 + x1 <= 1.5, x0, x1 ∈ {0,1}
    Optimal: x0=1, x1=0 (or x0=0, x1=1), obj=-1 (minimising -x0-x1)
    """
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.5],
        integer_vars=[0, 1],
        ub=[1.0, 1.0],
    )
    result = solve_milp(prob)
    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert abs(result.objective - (-1.0)) < 1e-4, f"obj={result.objective}"
    assert result.x is not None
    # Both vars should be binary
    assert abs(result.x[0] - round(result.x[0])) < 1e-4
    assert abs(result.x[1] - round(result.x[1])) < 1e-4


# ── 2. Classic 2-var MILP ─────────────────────────────────────────────────────

def test_milp_2var_classic():
    """
    max 3x0 + 5x1  s.t. x0 <= 4, 2x1 <= 12, 3x0+2x1 <= 18
        x0, x1 ∈ Z≥0
    LP optimal: x0=2, x1=6, obj=36 (integer)
    MILP optimal same as LP here since LP solution is integer.
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
        integer_vars=[0, 1],
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert abs(result.objective - (-36.0)) < 1e-4, f"obj={result.objective}"
    assert abs(result.x[0] - round(result.x[0])) < 1e-4
    assert abs(result.x[1] - round(result.x[1])) < 1e-4


# ── 3. MILP with fractional LP optimum ───────────────────────────────────────

def test_milp_fractional_lp_optimum():
    """
    min 2x0 + x1
    s.t.  x0 + x1 >= 3.5   (written as -x0 - x1 <= -3.5)
          x0, x1 ∈ Z≥0

    LP relaxation optimal: x0=0, x1=3.5, obj=3.5
    MILP optimal:          x0=0, x1=4,   obj=4
    """
    prob = make_problem(
        c=[2.0, 1.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-3.5],
        integer_vars=[0, 1],
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 4.0) < 1e-4, f"MILP obj={result.objective}, expected 4"
    # Verify integrality
    for j in [0, 1]:
        assert abs(result.x[j] - round(result.x[j])) < 1e-4


# ── 4. Infeasible MILP ────────────────────────────────────────────────────────

def test_milp_infeasible():
    """
    x0 + x1 <= -1  (infeasible since x0, x1 >= 0)
    """
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[-1.0],
        integer_vars=[0, 1],
    )
    result = solve_milp(prob)
    assert result.status == "infeasible"


# ── 5. Pure LP (no integer vars) ──────────────────────────────────────────────

def test_milp_no_integer_vars():
    """
    When no variables are integer, MILP reduces to LP.
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
        integer_vars=[],
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert abs(result.objective - (-36.0)) < 1e-4


# ── 6. Mixed-integer (some continuous, some integer) ─────────────────────────

def test_milp_mixed_integer():
    """
    min 3x0 + 2x1
    s.t. x0 + x1 >= 5   (-x0 - x1 <= -5)
         x0 >= 0, x1 >= 0
         x0 ∈ Z  (only x0 is integer)

    LP optimal: x0=0, x1=5, obj=10
    MILP optimal: same (x0=0 is already integer)
    """
    prob = make_problem(
        c=[3.0, 2.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-5.0],
        integer_vars=[0],
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert result.objective <= 10.0 + 1e-4
    # x0 must be integer
    assert abs(result.x[0] - round(result.x[0])) < 1e-4


# ── 7. Brute-force cross-check on small 0-1 instance ─────────────────────────

def test_milp_brute_force_binary():
    """
    Brute-force all 2^4 = 16 binary assignments for a 4-variable knapsack,
    compare to B&B result.
    """
    rng = np.random.default_rng(99)
    n = 4
    weights = rng.uniform(1, 5, n)
    values  = rng.uniform(1, 10, n)
    capacity = weights.sum() * 0.6  # ~60% of full capacity

    # Maximise sum(value * x)  →  minimise -sum(value * x)
    c = -values
    A_ub = [weights.tolist()]
    b_ub = [capacity]

    prob = make_problem(c=c.tolist(), A_ub=A_ub, b_ub=b_ub,
                        integer_vars=list(range(n)), ub=[1.0]*n)
    milp_result = solve_milp(prob)

    # Brute force
    best_obj = np.inf
    for mask in range(2**n):
        x_bf = np.array([(mask >> j) & 1 for j in range(n)], dtype=float)
        if weights @ x_bf <= capacity + 1e-6:
            obj = float(c @ x_bf)
            if obj < best_obj:
                best_obj = obj

    assert milp_result.status == "optimal"
    assert abs(milp_result.objective - best_obj) < 1e-4, (
        f"B&B={milp_result.objective:.4f}, brute-force={best_obj:.4f}"
    )


# ── 8. Root LP relaxation bound ───────────────────────────────────────────────

def test_milp_lp_relaxation_bound():
    """
    LP relaxation must be ≤ MILP optimal (for minimisation).
    """
    prob = make_problem(
        c=[2.0, 1.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-3.5],
        integer_vars=[0, 1],
    )
    result = solve_milp(prob)
    assert result.status == "optimal"
    assert result.lp_relaxation <= result.objective + 1e-6, (
        f"LP bound {result.lp_relaxation} > MILP obj {result.objective}"
    )
