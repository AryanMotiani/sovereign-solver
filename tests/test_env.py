"""
tests/test_env.py
-----------------
Phase 1A Gate — environment sanity check.

Verifies:
  1. numpy, scipy, highspy are importable
  2. HiGHS can solve a trivial 1-variable LP correctly
  3. is_feasible() works correctly on trivial feasible/infeasible cases
  4. Problem dataclass validates correctly
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solver.problem import Problem
from solver.utils.feasibility import is_feasible


# ── 1. Import checks ─────────────────────────────────────────────────────────

def test_numpy_importable():
    import numpy
    assert numpy.__version__

def test_scipy_importable():
    import scipy
    assert scipy.__version__

def test_highspy_importable():
    import highspy
    # highspy should be importable
    assert highspy is not None


# ── 2. HiGHS baseline sanity ─────────────────────────────────────────────────

def test_highspy_solves_1var_lp():
    """
    Solve  min -x  s.t. x <= 5, x >= 0  via highspy.
    Expected optimal: x=5, obj=-5.
    """
    import highspy
    h = highspy.Highs()
    h.silent()  # suppress output

    # Add variable: 0 <= x <= inf, objective coeff = -1
    h.addVar(0.0, 1e30)
    h.changeColCost(0, -1.0)

    # Add constraint: x <= 5
    h.addRow(-1e30, 5.0, 1, [0], [1.0])

    h.run()

    info = h.getInfoValue("primal_solution_status")
    # info is (HighsStatus, value) — check value
    sol = h.getSolution()
    obj = h.getInfoValue("objective_function_value")[1]

    assert abs(obj - (-5.0)) < 1e-6, f"HiGHS 1-var LP: expected obj=-5, got {obj}"
    assert abs(sol.col_value[0] - 5.0) < 1e-6


# ── 3. is_feasible checks ────────────────────────────────────────────────────

def _make_simple_problem():
    """
    Simple LP:  min x1 + x2
    s.t.  x1 + x2 <= 10
          x1, x2 >= 0, <= 6
    """
    return Problem(
        c=np.array([1.0, 1.0]),
        A_ub=sp.csr_matrix(np.array([[1.0, 1.0]])),
        b_ub=np.array([10.0]),
        A_eq=sp.csr_matrix((0, 2), dtype=float),
        b_eq=np.zeros(0),
        lb=np.array([0.0, 0.0]),
        ub=np.array([6.0, 6.0]),
        integer_mask=np.array([False, False]),
    )

def test_is_feasible_returns_true_for_valid_point():
    prob = _make_simple_problem()
    x = np.array([3.0, 4.0])  # 3+4=7 <= 10, within bounds
    assert is_feasible(prob, x)

def test_is_feasible_returns_false_ub_violation():
    prob = _make_simple_problem()
    x = np.array([7.0, 1.0])  # x1=7 > ub=6
    assert not is_feasible(prob, x)

def test_is_feasible_returns_false_ineq_violation():
    prob = _make_simple_problem()
    x = np.array([6.0, 5.0])  # 6+5=11 > 10
    assert not is_feasible(prob, x)

def test_is_feasible_returns_false_lb_violation():
    prob = _make_simple_problem()
    x = np.array([-1.0, 3.0])  # x1 < 0
    assert not is_feasible(prob, x)

def test_is_feasible_equality_constraint():
    prob = Problem(
        c=np.array([1.0, 1.0]),
        A_ub=sp.csr_matrix((0, 2), dtype=float),
        b_ub=np.zeros(0),
        A_eq=sp.csr_matrix(np.array([[1.0, 1.0]])),
        b_eq=np.array([5.0]),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.zeros(2, dtype=bool),
    )
    assert is_feasible(prob, np.array([2.0, 3.0]))      # 2+3=5 ✓
    assert not is_feasible(prob, np.array([2.0, 2.5]))  # 2+2.5=4.5 ≠ 5

def test_is_feasible_integer_constraint():
    prob = Problem(
        c=np.array([1.0, 1.0]),
        A_ub=sp.csr_matrix((0, 2), dtype=float),
        b_ub=np.zeros(0),
        A_eq=sp.csr_matrix((0, 2), dtype=float),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        integer_mask=np.array([True, False]),
    )
    assert is_feasible(prob, np.array([2.0, 3.7]))      # x1=2 is integer ✓
    assert not is_feasible(prob, np.array([2.5, 3.7]))  # x1=2.5 not integer ✗

def test_is_feasible_shape_mismatch():
    prob = _make_simple_problem()
    assert not is_feasible(prob, np.array([1.0, 2.0, 3.0]))  # wrong shape


# ── 4. Problem dataclass validation ──────────────────────────────────────────

def test_problem_validates_ok():
    prob = _make_simple_problem()
    prob.validate()  # Should not raise

def test_problem_validates_dimension_mismatch():
    with pytest.raises(ValueError):
        prob = Problem(
            c=np.array([1.0, 1.0]),
            A_ub=sp.csr_matrix(np.array([[1.0, 1.0, 1.0]])),  # wrong: 3 cols, n=2
            b_ub=np.array([10.0]),
            A_eq=sp.csr_matrix((0, 2), dtype=float),
            b_eq=np.zeros(0),
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
            integer_mask=np.zeros(2, dtype=bool),
        )
        prob.validate()

def test_problem_eval_obj():
    prob = _make_simple_problem()
    x = np.array([3.0, 4.0])
    assert abs(prob.eval_obj(x) - 7.0) < 1e-12

def test_problem_repr():
    prob = _make_simple_problem()
    r = repr(prob)
    assert "LP" in r
    assert "n_vars=2" in r
