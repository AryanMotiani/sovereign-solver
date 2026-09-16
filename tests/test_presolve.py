"""
tests/test_presolve.py
-----------------------
Test suite for Phase 2A: Presolve + Postsolve.

Gate T-06: pytest tests/test_presolve.py → all pass
          (zero objective mismatches on presolved vs. direct solve)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.lp.simplex_revised import solve_lp_revised
from solver.presolve.postsolve import postsolve
from solver.presolve.presolve import presolve
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, A_eq=None, b_eq=None, lb=None, ub=None):
    n = len(c)
    m_ub = len(b_ub)
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix(np.array(A_eq, dtype=float)),
        b_eq=np.array(b_eq, dtype=float),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        integer_mask=np.zeros(n, dtype=bool),
    )


# ── 1. Fixed variable elimination ────────────────────────────────────────────

def test_presolve_fixed_variable():
    """
    Variable x0 is fixed: lb=ub=2.  Presolve should eliminate it and
    adjust the RHS.  Postsolve should restore x0=2.
    """
    # min x0 + x1  s.t. x0 + x1 >= 3 (written as -x0 - x1 <= -3)
    # with x0 fixed at 2 → reduced: min x1 s.t. x1 >= 1
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-3.0],
        lb=[2.0, 0.0],
        ub=[2.0, np.inf],
    )
    result = presolve(prob)
    assert not result.infeasible
    assert result.n_fixed >= 1, "Expected at least one fixed variable"

    # Solve reduced problem
    red_result = solve_lp_revised(result.problem)
    assert red_result.status == "optimal"

    # Postsolve
    x_full = postsolve(result, red_result.x)
    assert len(x_full) == 2
    assert abs(x_full[0] - 2.0) < 1e-5, f"x0 should be fixed at 2, got {x_full[0]}"
    assert abs(x_full[1] - 1.0) < 1e-4, f"x1 should be 1, got {x_full[1]}"


# ── 2. Empty row removal ──────────────────────────────────────────────────────

def test_presolve_removes_empty_rows():
    """
    A row that becomes all-zeros after bound shifts should be removed.
    """
    # min x  s.t. 0*x <= 5 (trivially redundant), x >= 0
    prob = make_problem(
        c=[1.0],
        A_ub=[[0.0], [1.0]],
        b_ub=[5.0, 3.0],  # first row redundant, second: x <= 3
    )
    result = presolve(prob)
    assert not result.infeasible
    # Reduced problem should have fewer rows
    assert result.problem.n_ineq <= 1


def test_presolve_detects_infeasible_empty_row():
    """
    A row 0x <= -1 is infeasible.  Presolve should detect and flag it.
    """
    prob = make_problem(
        c=[1.0],
        A_ub=[[0.0]],
        b_ub=[-1.0],
    )
    result = presolve(prob)
    assert result.infeasible, "Expected infeasibility to be detected"


# ── 3. Row singleton (equality) ───────────────────────────────────────────────

def test_presolve_row_singleton_equality():
    """
    Equality 3x1 = 9 → x1 is fixed to 3.
    """
    # min x0 + x1  s.t. 3x1 = 9, x0 + x1 <= 10
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[10.0],
        A_eq=[[0.0, 3.0]],
        b_eq=[9.0],
    )
    result = presolve(prob)
    assert not result.infeasible
    assert result.n_fixed >= 1

    red_result = solve_lp_revised(result.problem)
    assert red_result.status == "optimal"
    x_full = postsolve(result, red_result.x)
    assert abs(x_full[1] - 3.0) < 1e-4, f"x1 should be fixed to 3, got {x_full[1]}"


# ── 4. Bound tightening ───────────────────────────────────────────────────────

def test_presolve_bound_tightening():
    """
    Single-variable inequality row x0 <= 5 should tighten ub from inf to 5.
    """
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[1.0, 0.0]],   # x0 <= 5
        b_ub=[5.0],
    )
    result = presolve(prob)
    assert not result.infeasible
    # Bound tightening should have been applied to x0
    assert result.n_rows_removed >= 1 or result.n_bound_tightened >= 1


# ── 5. Obj match: presolve + postsolve vs. direct solve ───────────────────────

@pytest.mark.parametrize("c,A_ub,b_ub,A_eq,b_eq", [
    # Case A: all-inequalities, no presolve trigger
    ([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18], None, None),
    # Case B: fixed variable reduces problem
    ([1, 1, 2], [[1, 1, 0]], [4], [[0, 0, 1]], [3]),  # 3rd var fixed by eq
    # Case C: equality row singleton
    ([1, 2], [[1, 1]], [5], [[2, 0]], [4]),  # x0 fixed to 2
])
def test_presolve_obj_matches_direct(c, A_ub, b_ub, A_eq, b_eq):
    """
    After presolve → solve → postsolve, the objective must equal the
    objective of directly solving the original problem.
    """
    prob = make_problem(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    # Direct solve
    direct = solve_lp_revised(prob)

    # Presolved solve
    pres = presolve(prob)
    if pres.infeasible:
        assert direct.status == "infeasible"
        return

    red = solve_lp_revised(pres.problem)
    if red.status != "optimal":
        assert direct.status != "optimal"
        return

    x_full = postsolve(pres, red.x)
    obj_presolved = float(np.array(c) @ x_full)

    assert abs(obj_presolved - direct.objective) < 1e-4, (
        f"Obj mismatch: presolved={obj_presolved:.6g} vs direct={direct.objective:.6g}"
    )


def test_presolve_multi_variable_bound_tightening():
    """
    Implied bound tightening:
    2*x0 + 3*x1 <= 6 with x0, x1 >= 0 implies x0 <= 3 and x1 <= 2.
    """
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[2.0, 3.0]],
        b_ub=[6.0],
    )
    result = presolve(prob)
    assert not result.infeasible
    # x0 should have ub <= 3.0 + 1e-5, x1 should have ub <= 2.0 + 1e-5
    assert result.problem.ub[0] <= 3.0 + 1e-4
    assert result.problem.ub[1] <= 2.0 + 1e-4


def test_presolve_parallel_rows():
    """
    Parallel row detection:
    Row 1: x0 + x1 <= 4
    Row 2: 2*x0 + 2*x1 <= 10 (which is x0 + x1 <= 5, redundant)
    Row 2 should be eliminated.
    """
    prob = make_problem(
        c=[1.0, 1.0],
        A_ub=[[1.0, 1.0], [2.0, 2.0]],
        b_ub=[4.0, 10.0],
    )
    result = presolve(prob)
    assert not result.infeasible
    assert result.problem.n_ineq == 1
    assert result.problem.b_ub[0] == 4.0


def test_presolve_probing_binary():
    """
    Binary variable probing:
    x0, x1 in {0, 1}
    x0 + x1 <= 1
    -x0 - 2*x1 <= -1.5 (i.e. x0 + 2*x1 >= 1.5)
    If x1 = 0: x0 >= 1.5 (impossible since x0 in {0, 1}) -> x1 must be 1!
    When x1 = 1: x0 + 1 <= 1 -> x0 = 0.
    """
    prob = Problem(
        c=np.array([1.0, 1.0]),
        A_ub=sp.csr_matrix([[1.0, 1.0], [-1.0, -2.0]]),
        b_ub=np.array([1.0, -1.5]),
        A_eq=sp.csr_matrix((0, 2)),
        b_eq=np.zeros(0),
        lb=np.zeros(2),
        ub=np.ones(2),
        integer_mask=np.array([True, True]),
    )
    result = presolve(prob)
    assert not result.infeasible
    # At least one variable should be fixed by probing
    assert result.n_fixed >= 1



# ── 6. No-op presolve (trivial problem) ──────────────────────────────────────

def test_presolve_noop_passthrough():
    """
    A problem with no fixable variables/rows should pass through unchanged
    (or with minor row removal).
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
    )
    result = presolve(prob)
    assert not result.infeasible
    # Problem may have same or fewer constraints; objective must be preserved
    direct = solve_lp_revised(prob)
    red = solve_lp_revised(result.problem)
    assert red.status == "optimal"
    x_full = postsolve(result, red.x)
    obj = float(np.array([-3.0, -5.0]) @ x_full[:2])
    assert abs(obj - direct.objective) < 1e-4
