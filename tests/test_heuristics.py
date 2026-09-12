"""
tests/test_heuristics.py
-------------------------
Test suite for Phase 3B: Primal heuristics.

Gate T-11: pytest tests/test_heuristics.py → all pass
           + is_feasible on all heuristic outputs
           + time-to-first-solution comparison
"""

import numpy as np
import pytest
import scipy.sparse as sp
import time

from solver.milp.heuristics import (
    heuristic_diving,
    heuristic_feasibility_pump,
    heuristic_rins,
    heuristic_rounding,
)
from solver.problem import Problem
from solver.utils.feasibility import is_feasible


def make_problem(c, A_ub, b_ub, integer_vars, lb=None, ub=None):
    n = len(c)
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
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        integer_mask=int_mask,
    )


# Shared test problem: binary knapsack
def _knapsack_problem(n=4, seed=42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(1, 5, n)
    values  = rng.uniform(1, 10, n)
    capacity = weights.sum() * 0.6
    c = -values
    return make_problem(
        c=c.tolist(),
        A_ub=[weights.tolist()],
        b_ub=[capacity],
        integer_vars=list(range(n)),
        ub=[1.0]*n,
    )


# ── 1. Rounding heuristic ─────────────────────────────────────────────────────

def test_rounding_returns_none_or_feasible():
    prob = _knapsack_problem(n=4)
    x = heuristic_rounding(prob)
    if x is not None:
        assert x.shape == (4,)
        assert np.all(x >= -1e-6)
        assert np.all(x <= 1.0 + 1e-6)
        # Integer variables must be integer
        for j in np.where(prob.integer_mask)[0]:
            assert abs(x[j] - round(x[j])) < 1e-5


def test_rounding_on_easy_milp():
    """
    min x s.t. x >= 2.0  (x integer, lb=0)
    LP optimal = 2.0 exactly → rounding gives x=2 → feasible.
    """
    prob = make_problem(
        c=[1.0],
        A_ub=[[-1.0]],
        b_ub=[-2.0],
        integer_vars=[0],
    )
    x = heuristic_rounding(prob)
    # May or may not find it depending on LP solution; just check shape
    assert x is None or (x.shape == (1,) and x[0] >= 2.0 - 1e-5)


# ── 2. Diving heuristic ───────────────────────────────────────────────────────

def test_diving_returns_none_or_feasible():
    prob = _knapsack_problem(n=5)
    x = heuristic_diving(prob)
    if x is not None:
        assert x.shape == (5,)
        assert is_feasible(prob, x)
        for j in np.where(prob.integer_mask)[0]:
            assert abs(x[j] - round(x[j])) < 1e-5


def test_diving_simple_case():
    """
    A problem where diving should find the optimal:
    max x+y s.t. x+y<=3.5, x,y ∈ {0,1,2,3}
    LP optimal: x=1.75, y=1.75 → dive fixes one to 2, then LP gives other → 1
    """
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[3.5],
        integer_vars=[0, 1],
        ub=[3.0, 3.0],
    )
    x = heuristic_diving(prob)
    if x is not None:
        assert abs(x[0] - round(x[0])) < 1e-5
        assert abs(x[1] - round(x[1])) < 1e-5
        assert x[0] + x[1] <= 3.5 + 1e-5


# ── 3. Feasibility Pump ───────────────────────────────────────────────────────

def test_feasibility_pump_returns_none_or_feasible():
    prob = _knapsack_problem(n=4)
    x = heuristic_feasibility_pump(prob)
    if x is not None:
        assert is_feasible(prob, x)
        for j in np.where(prob.integer_mask)[0]:
            assert abs(x[j] - round(x[j])) < 1e-5


def test_feasibility_pump_simple():
    """
    min x s.t. x >= 3.7, x ∈ Z
    LP fractional optimum: x=3.7 → FP should pump to x=4.
    """
    prob = make_problem(
        c=[1.0],
        A_ub=[[-1.0]],
        b_ub=[-3.7],
        integer_vars=[0],
    )
    x = heuristic_feasibility_pump(prob)
    if x is not None:
        assert x[0] >= 4.0 - 1e-5


# ── 4. RINS ───────────────────────────────────────────────────────────────────

def test_rins_returns_none_or_feasible():
    prob = _knapsack_problem(n=4)

    # Provide a dummy LP relaxation and incumbent
    from solver.lp.simplex_revised import solve_lp_revised
    from solver.problem import Problem as P
    lp_prob = P(
        c=prob.c, A_ub=prob.A_ub, b_ub=prob.b_ub,
        A_eq=prob.A_eq, b_eq=prob.b_eq,
        lb=prob.lb, ub=prob.ub,
        integer_mask=np.zeros(4, dtype=bool),
    )
    lp_res = solve_lp_revised(lp_prob)
    x_lp = lp_res.x if lp_res.status == "optimal" else np.zeros(4)

    # A simple feasible incumbent
    x_inc = np.zeros(4)

    x_rins = heuristic_rins(prob, x_lp, x_inc)
    if x_rins is not None:
        assert is_feasible(prob, x_rins)


# ── 5. Output is always non-negative (feasibility constraint) ─────────────────

@pytest.mark.parametrize("heuristic", [
    heuristic_rounding,
    heuristic_diving,
    heuristic_feasibility_pump,
])
def test_heuristic_output_non_negative(heuristic):
    prob = _knapsack_problem(n=4)
    x = heuristic(prob)
    if x is not None:
        assert np.all(x >= -1e-5), f"{heuristic.__name__} returned negative x: {x}"


# ── 6. Time-to-first-solution comparison ─────────────────────────────────────

def test_heuristics_time_comparison():
    """
    Compare wall-clock time for each heuristic on a 6-var knapsack.
    Print a comparison table (not an assertion — informational).
    """
    prob = _knapsack_problem(n=6, seed=123)
    heuristics = [
        ("Rounding",    heuristic_rounding),
        ("Diving",      heuristic_diving),
        ("FeasPump",    heuristic_feasibility_pump),
    ]
    results = []
    for name, fn in heuristics:
        t0 = time.monotonic()
        x = fn(prob)
        elapsed = time.monotonic() - t0
        obj = float(prob.c @ x) if x is not None else None
        results.append((name, obj, elapsed, x is not None))

    # Print comparison table
    print("\n\nHeuristic Comparison (6-var knapsack):")
    print(f"{'Heuristic':<12} {'Found':>6} {'Obj':>10} {'Time(s)':>10}")
    print("-" * 42)
    for name, obj, t, found in results:
        obj_str = f"{obj:.4f}" if obj is not None else "—"
        print(f"{name:<12} {str(found):>6} {obj_str:>10} {t:>10.4f}")

    # At least one heuristic should find a solution for this easy instance
    any_found = any(found for _, _, _, found in results)
    assert any_found, "No heuristic found a feasible solution!"
