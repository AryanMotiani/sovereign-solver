"""
tests/test_cuts.py
------------------
Test suite for Phase 3A: Cutting plane generators.

Gate T-10: pytest tests/test_cuts.py → all pass
           + validity on toy instances verified
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.milp.cuts import (
    add_cuts_to_problem,
    build_conflict_graph,
    generate_all_cuts,
    generate_clique_cuts,
    generate_cover_cuts,
    generate_mir_cuts,
)
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, integer_vars, lb=None, ub=None):
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.ones(n)
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


# ── 1. MIR cuts validity ──────────────────────────────────────────────────────

def test_mir_cuts_are_valid_for_integer_solutions():
    """
    Every MIR cut aᵀx ≤ b must hold for all integer-feasible solutions
    of the original problem.  We check against the LP optimal and a
    handful of integer solutions.
    """
    # Knapsack: max x0+x1+x2 s.t. 3x0 + 4x1 + 2x2 <= 5, x ∈ {0,1}
    prob = make_problem(
        c=[-1.0, -1.0, -1.0],
        A_ub=[[3.0, 4.0, 2.0]],
        b_ub=[5.0],
        integer_vars=[0, 1, 2],
    )
    # LP relaxation point (fractional)
    x_star = np.array([0.5, 0.5, 0.5])
    cuts = generate_mir_cuts(prob, x_star)

    # Verify every cut is satisfied by all integer-feasible solutions
    int_solutions = [
        np.array([1.0, 0.0, 1.0]),  # feasible: 3+2=5 ≤ 5 ✓
        np.array([0.0, 1.0, 0.0]),  # feasible: 4 ≤ 5 ✓
        np.array([0.0, 0.0, 1.0]),  # feasible: 2 ≤ 5 ✓
        np.array([1.0, 0.0, 0.0]),  # feasible: 3 ≤ 5 ✓
    ]
    for a_cut, b_cut in cuts:
        for x_int in int_solutions:
            val = float(a_cut @ x_int)
            assert val <= b_cut + 1e-6, (
                f"MIR cut violated by integer solution: a@x={val:.4f} > b={b_cut:.4f}"
            )


def test_mir_cuts_separate_fractional_point():
    """
    If a fractional point violates a MIR cut, the cut must separate it.
    """
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[3.0, 2.0]],
        b_ub=[3.7],
        integer_vars=[0, 1],
    )
    # LP fractional optimum: x* = (1.0, 0.35) → 3+0.7=3.7 tight
    x_star = np.array([1.0, 0.35])
    cuts = generate_mir_cuts(prob, x_star)

    # Each returned cut must be violated by x_star (that's the selection criterion)
    for a_cut, b_cut in cuts:
        violation = float(a_cut @ x_star) - b_cut
        assert violation > 1e-5, "MIR cut returned that doesn't separate x_star"


def test_mir_no_cuts_at_integer_point():
    """
    At an integer-feasible solution, no MIR cut should be violated.
    """
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[2.0],
        integer_vars=[0, 1],
    )
    x_int = np.array([1.0, 1.0])  # feasible integer point
    cuts = generate_mir_cuts(prob, x_int)
    for a_cut, b_cut in cuts:
        assert float(a_cut @ x_int) <= b_cut + 1e-4, "Cut violated at integer point!"


# ── 2. Cover cuts validity ────────────────────────────────────────────────────

def test_cover_cuts_are_valid():
    """
    Cover cuts must hold for all 0-1 feasible solutions.
    """
    prob = make_problem(
        c=[-3.0, -5.0, -2.0],
        A_ub=[[2.0, 3.0, 1.0]],
        b_ub=[4.0],
        integer_vars=[0, 1, 2],
    )
    x_star = np.array([0.7, 0.7, 0.7])  # fractional point
    cuts = generate_cover_cuts(prob, x_star)

    # All 0-1 feasible solutions
    feasible = []
    for mask in range(8):
        x = np.array([(mask >> j) & 1 for j in range(3)], dtype=float)
        if np.array([2.0, 3.0, 1.0]) @ x <= 4.0 + 1e-6:
            feasible.append(x)

    for a_cut, b_cut in cuts:
        for xf in feasible:
            assert float(a_cut @ xf) <= b_cut + 1e-6, (
                f"Cover cut violated at feasible integer solution"
            )


def test_cover_cuts_separate():
    """
    Cover cuts must be violated by the fractional input point (or at least
    not all satisfied strictly).
    """
    prob = make_problem(
        c=[-1.0, -1.0, -1.0],
        A_ub=[[3.0, 4.0, 2.0]],
        b_ub=[5.0],
        integer_vars=[0, 1, 2],
    )
    x_star = np.array([0.8, 0.7, 0.6])
    cuts = generate_cover_cuts(prob, x_star)
    for a_cut, b_cut in cuts:
        violation = float(a_cut @ x_star) - b_cut
        assert violation > 1e-5, "Cover cut doesn't separate x_star"


# ── 3. Clique cuts ────────────────────────────────────────────────────────────

def test_conflict_graph_detects_packing_row():
    """
    A packing row x0 + x1 + x2 <= 1 should generate all 3 conflict edges.
    """
    prob = make_problem(
        c=[-1.0, -1.0, -1.0],
        A_ub=[[1.0, 1.0, 1.0]],
        b_ub=[1.0],
        integer_vars=[0, 1, 2],
    )
    x_star = np.array([0.4, 0.4, 0.4])
    edges = build_conflict_graph(prob, x_star)
    edge_set = set(map(frozenset, edges))
    assert frozenset({0, 1}) in edge_set
    assert frozenset({0, 2}) in edge_set
    assert frozenset({1, 2}) in edge_set


def test_clique_cuts_validity():
    """
    Clique cuts sum_{j in clique} x_j <= 1 must hold for all
    integer-feasible solutions of a packing problem.
    """
    prob = make_problem(
        c=[-1.0, -1.0, -1.0],
        A_ub=[[1.0, 1.0, 1.0]],
        b_ub=[1.0],
        integer_vars=[0, 1, 2],
    )
    x_star = np.array([0.45, 0.45, 0.45])  # violates sum ≤ 1
    cuts = generate_clique_cuts(prob, x_star)

    # Feasible integer solutions: exactly one of them can be 1
    for a_cut, b_cut in cuts:
        for j in range(3):
            x_int = np.zeros(3)
            x_int[j] = 1.0
            assert float(a_cut @ x_int) <= b_cut + 1e-6


# ── 4. add_cuts_to_problem ────────────────────────────────────────────────────

def test_add_cuts_appends_rows():
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.5],
        integer_vars=[0, 1],
    )
    cuts = [(np.array([1.0, 0.0]), 0.5)]  # x0 <= 0.5
    prob2 = add_cuts_to_problem(prob, cuts)
    assert prob2.n_ineq == prob.n_ineq + 1


def test_add_cuts_empty_is_noop():
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.5],
        integer_vars=[0, 1],
    )
    prob2 = add_cuts_to_problem(prob, [])
    assert prob2.n_ineq == prob.n_ineq


# ── 5. generate_all_cuts smoke test ──────────────────────────────────────────

def test_generate_all_cuts_returns_list():
    prob = make_problem(
        c=[-1.0, -1.0, -1.0],
        A_ub=[[3.0, 4.0, 2.0]],
        b_ub=[5.0],
        integer_vars=[0, 1, 2],
    )
    x_star = np.array([0.6, 0.5, 0.7])
    cuts = generate_all_cuts(prob, x_star)
    assert isinstance(cuts, list)
    for a_cut, b_cut in cuts:
        assert len(a_cut) == 3
        assert isinstance(b_cut, float)
