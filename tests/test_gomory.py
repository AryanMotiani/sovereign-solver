"""
tests/test_gomory.py
--------------------
Tests for Gomory fractional cuts (solver/milp/gomory.py).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.milp.gomory import generate_gomory_cuts
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, integer_vars, lb=None, ub=None):
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, 10.0)
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
        sense="min",
    )


def test_gomory_cuts_separate_fractional():
    # Min -x0 - x1 s.t. 3x0 + 2x1 <= 3.7, x0, x1 >= 0 integer
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[3.0, 2.0]],
        b_ub=[3.7],
        integer_vars=[0, 1],
    )
    res = solve_lp_revised(prob)
    assert res.status == "optimal"
    x_star = res.x

    cuts = generate_gomory_cuts(prob, x_star, basis=res.basis)
    assert len(cuts) > 0, "Expected at least one Gomory cut"

    for a_cut, b_cut in cuts:
        violation = float(a_cut @ x_star) - b_cut
        assert violation > 1e-4, f"Cut does not separate x_star: violation={violation}"


def test_gomory_cuts_valid_for_integers():
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[3.0, 2.0]],
        b_ub=[3.7],
        integer_vars=[0, 1],
    )
    res = solve_lp_revised(prob)
    cuts = generate_gomory_cuts(prob, res.x, basis=res.basis)

    # Valid integer solutions within 3x0 + 2x1 <= 3.7
    int_feasible = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    for a_cut, b_cut in cuts:
        for x_int in int_feasible:
            val = float(a_cut @ x_int)
            assert val <= b_cut + 1e-5, f"Gomory cut cut off integer feasible point: {val} > {b_cut}"


def test_gomory_no_cuts_at_integer_solution():
    prob = make_problem(
        c=[-1.0, -1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[2.0],
        integer_vars=[0, 1],
    )
    # Integer point
    x_int = np.array([1.0, 1.0])
    cuts = generate_gomory_cuts(prob, x_int)
    assert len(cuts) == 0, "No cuts should be generated at integer solution"
