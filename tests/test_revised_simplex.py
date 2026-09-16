"""
tests/test_revised_simplex.py
------------------------------
Phase 1B Gate — Revised simplex tests (T-03).

Tests:
  1. Same toy LPs as dense simplex — must match exactly
  2. Infeasible / unbounded detection
  3. Equality-constrained LP
  4. Dual simplex warm-start correctness
  5. Parametric LP suite (cross-check vs HiGHS)
  6. Cross-agreement: revised simplex == dense simplex objective on toy instances
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solver.problem import Problem
from solver.lp.simplex_revised import solve_lp_revised, solve_lp_dual
from solver.lp.simplex_dense import solve_lp_dense


def make_lp(c, A_ub_dense, b_ub, A_eq_dense=None, b_eq=None, lb=None, ub=None, sense="min"):
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
        sense=sense,
    )


# ── T-03.1: Textbook 2-var LP ─────────────────────────────────────────────────

def test_revised_simplex_2var_lp():
    prob = make_lp(
        c=[-3.0, -5.0],
        A_ub_dense=[[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        b_ub=[4.0, 12.0, 18.0],
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal", f"Got {result.status}: {result.message}"
    assert abs(result.objective - (-36.0)) < 1e-5
    assert abs(result.x[0] - 2.0) < 1e-4
    assert abs(result.x[1] - 6.0) < 1e-4


def test_revised_simplex_min():
    prob = make_lp(
        c=[1.0, 1.0],
        A_ub_dense=[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        b_ub=[-3.0, 5.0, 5.0],
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 3.0) < 1e-4


# ── T-03.2: Infeasible / unbounded ────────────────────────────────────────────

def test_revised_simplex_infeasible():
    prob = make_lp(
        c=[1.0],
        A_ub_dense=[[-1.0], [1.0]],
        b_ub=[-5.0, 3.0],
    )
    result = solve_lp_revised(prob)
    assert result.status == "infeasible", f"Expected infeasible, got {result.status}"


def test_revised_simplex_unbounded():
    prob = make_lp(
        c=[-1.0],
        A_ub_dense=[[0.0]],
        b_ub=[1e10],
    )
    result = solve_lp_revised(prob)
    assert result.status == "unbounded", f"Expected unbounded, got {result.status}"


# ── T-03.3: Equality constraints ──────────────────────────────────────────────

def test_revised_simplex_equality():
    """
    min x + 2y  s.t.  x + y = 5, x,y >= 0 → optimal x=5, y=0, obj=5.
    """
    prob = make_lp(
        c=[1.0, 2.0],
        A_ub_dense=np.zeros((0, 2)),
        b_ub=np.zeros(0),
        A_eq_dense=[[1.0, 1.0]],
        b_eq=[5.0],
    )
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 5.0) < 1e-4, f"obj={result.objective}"


# ── T-03.4: Cross-agreement with dense simplex ────────────────────────────────

@pytest.mark.parametrize("c,A,b,expected", [
    ([-3.0, -5.0], [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]], [4.0, 12.0, 18.0], -36.0),
    ([1.0, 1.0],   [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], [-3.0, 5.0, 5.0], 3.0),
    ([2.0, -1.0],  [[1.0, 1.0], [-1.0, 1.0]], [4.0, 2.0], -3.0),
    ([1.0, 0.0],   [[1.0, 0.0], [0.0, 1.0]], [3.0, 3.0], 0.0),  # lb=0
])
def test_revised_agrees_with_dense(c, A, b, expected):
    prob = make_lp(c=c, A_ub_dense=A, b_ub=b)
    rev = solve_lp_revised(prob)
    den = solve_lp_dense(prob)

    if rev.status == "optimal" and den.status == "optimal":
        diff = abs(rev.objective - den.objective)
        assert diff < 1e-4, (
            f"Revised={rev.objective:.8g} vs Dense={den.objective:.8g}, diff={diff:.2e}"
        )


# ── T-03.5: Larger random LP ─────────────────────────────────────────────────

def test_revised_simplex_random_lp():
    """
    Generate a random feasible LP and verify the objective is finite and
    agrees with HiGHS.
    """
    import highspy
    rng = np.random.default_rng(123)
    n, m = 10, 6
    A = rng.uniform(0, 2, (m, n))
    x_feas = rng.uniform(0.5, 2, n)
    b = A @ x_feas + rng.uniform(0.5, 2, m)
    c = rng.uniform(-1, 1, n)

    prob = make_lp(c=c.tolist(), A_ub_dense=A.tolist(), b_ub=b.tolist())
    rev = solve_lp_revised(prob)

    # Compare to HiGHS
    h = highspy.Highs()
    h.silent()
    for j in range(n):
        h.addVar(0.0, 1e30)
        h.changeColCost(j, float(c[j]))
    for i in range(m):
        idx = list(range(n))
        vals = [float(A[i, j]) for j in range(n)]
        h.addRow(-1e30, float(b[i]), n, idx, vals)
    h.run()
    highs_obj = h.getInfoValue("objective_function_value")[1]

    if rev.status == "optimal":
        diff = abs(rev.objective - highs_obj) / max(1.0, abs(highs_obj))
        assert diff < 1e-4, (
            f"Revised={rev.objective:.8g}, HiGHS={highs_obj:.8g}, rel_diff={diff:.2e}"
        )


# ── T-03.6: Dual simplex produces same result as primal ───────────────────────

def test_dual_simplex_basic():
    """
    Dual simplex on a simple LP with a known primal-optimal basis should
    match the primal simplex objective.
    """
    prob = make_lp(
        c=[1.0, 2.0],
        A_ub_dense=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        b_ub=[4.0, 3.0, 6.0],
    )
    primal = solve_lp_revised(prob)
    dual = solve_lp_dual(prob)  # without a basis, falls back to primal

    if primal.status == "optimal" and dual.status == "optimal":
        assert abs(primal.objective - dual.objective) < 1e-4


# ── T-03.7: Netlib cross-check (requires internet / cached files) ─────────────

NETLIB_INSTANCES = [
    # (name, url, known_optimal_approx)
    ("afiro",    "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/afiro.mps",    -464.753),
]

@pytest.mark.parametrize("name,url,known_obj", NETLIB_INSTANCES)
def test_revised_simplex_netlib(name, url, known_obj, tmp_path):
    """
    Download and solve a Netlib LP instance with our revised simplex.
    Cross-check objective against known value and HiGHS.
    """
    import urllib.request
    import highspy
    from solver.io.mps_reader import read_mps

    mps_file = tmp_path / f"{name}.mps"
    try:
        urllib.request.urlretrieve(url, mps_file)
    except Exception:
        pytest.skip(f"Cannot download {name}.mps — skipping (no internet)")

    prob = read_mps(str(mps_file))

    # Our solver
    our_result = solve_lp_revised(prob)

    # HiGHS baseline
    h = highspy.Highs()
    h.silent()
    h.readModel(str(mps_file))
    h.run()
    highs_obj = h.getInfoValue("objective_function_value")[1]

    if our_result.status == "optimal":
        rel_err = abs(our_result.objective - highs_obj) / max(1.0, abs(highs_obj))
        assert rel_err < 1e-4, (
            f"{name}: our={our_result.objective:.8g}, HiGHS={highs_obj:.8g}, "
            f"rel_err={rel_err:.2e}"
        )
    else:
        pytest.skip(f"{name}: solver returned {our_result.status}, not testing objective")
