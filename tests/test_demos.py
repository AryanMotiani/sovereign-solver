"""
tests/test_demos.py
--------------------
Test suite for Phase 4B domain demos.

Gate T-13: pytest tests/test_demos.py → all pass
           + result is feasible and objective is finite
"""

import numpy as np
import pytest

from demos.crude_blend_lp import build_blend_problem
from demos.refinery_scheduling_milp import build_schedule_problem
from solver.lp.simplex_revised import solve_lp_revised
from solver.milp.branch_and_bound import solve_milp
from solver.utils.feasibility import is_feasible


# ── Demo 1: Crude blend LP ────────────────────────────────────────────────────

def test_crude_blend_lp_builds():
    prob = build_blend_problem()
    assert prob.n_vars == 3
    assert prob.n_ineq == 3  # sulphur + min diesel + capacity


def test_crude_blend_lp_solves():
    prob = build_blend_problem()
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    assert np.isfinite(result.objective)
    assert result.x is not None
    assert result.x.shape == (3,)


def test_crude_blend_lp_feasibility():
    prob = build_blend_problem()
    result = solve_lp_revised(prob)
    assert result.status == "optimal"
    # Solution must be non-negative
    assert np.all(result.x >= -1e-5), f"Negative crude volumes: {result.x}"
    # Total throughput ≤ 10,000
    assert result.x.sum() <= 10_000 + 1e-4
    # Diesel ≥ 3,000 kL/day
    from demos.crude_blend_lp import DIESEL_YIELD, MIN_DIESEL
    diesel_vol = DIESEL_YIELD @ result.x
    assert diesel_vol >= MIN_DIESEL - 1e-4, f"Diesel {diesel_vol:.0f} < {MIN_DIESEL}"
    # Sulphur ≤ 1.5%
    from demos.crude_blend_lp import SULPHUR_PCT, MAX_SULPHUR
    throughput = result.x.sum()
    sulphur = (SULPHUR_PCT @ result.x) / max(throughput, 1e-9)
    assert sulphur <= MAX_SULPHUR + 1e-4, f"Sulphur {sulphur:.3f} > {MAX_SULPHUR}"


# ── Demo 2: Refinery scheduling MILP ─────────────────────────────────────────

def test_refinery_schedule_builds():
    prob = build_schedule_problem()
    assert prob.n_vars == 30          # 20 continuous + 10 binary
    assert prob.integer_mask.sum() == 10
    assert prob.n_ineq > 0


def test_refinery_schedule_solves():
    prob = build_schedule_problem()
    result = solve_milp(prob, time_limit=60.0)
    assert result.status in ("optimal", "node_limit", "time_limit")
    assert result.x is not None
    assert result.x.shape == (30,)
    assert np.isfinite(result.objective)


def test_refinery_schedule_binary_vars():
    prob = build_schedule_problem()
    result = solve_milp(prob, time_limit=60.0)
    assert result.status in ("optimal", "node_limit")
    x = result.x
    # Binary vars (FCC and HDS on/off) must be 0 or 1
    n_x = 20
    y = x[n_x:]
    for j, yj in enumerate(y):
        assert abs(yj - round(yj)) < 1e-4, f"y[{j}]={yj} not binary"


def test_refinery_schedule_throughput_non_negative():
    prob = build_schedule_problem()
    result = solve_milp(prob, time_limit=60.0)
    assert result.status in ("optimal", "node_limit")
    # All throughput variables ≥ 0
    assert np.all(result.x[:20] >= -1e-4), "Negative throughput found"


def test_refinery_schedule_lp_bound():
    """LP relaxation must be ≤ MILP optimal."""
    prob = build_schedule_problem()
    result = solve_milp(prob, time_limit=60.0)
    if result.status in ("optimal", "node_limit"):
        assert result.lp_relaxation <= result.objective + 1e-4, (
            f"LP bound {result.lp_relaxation:.2f} > MILP obj {result.objective:.2f}"
        )
