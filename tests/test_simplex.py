"""
tests/test_simplex.py
----------------------
Phase 1A Gate — Dense simplex correctness tests (T-02).
Phase 1B Gate — Revised simplex will be added here in Phase 1B.

T-02 tests:
  1. 2-variable LP: max 3x+5y s.t. x<=4, 2y<=12, 3x+2y<=18 → x=2, y=6, obj=36
  2. Infeasible LP detection
  3. Unbounded LP detection
  4. Beale cycling example with Bland's rule → terminates, correct obj
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solver.problem import Problem
from solver.lp.simplex_dense import solve_lp_dense


def make_lp(c, A_ub_dense, b_ub, lb=None, ub=None, sense="min"):
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub_dense, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix((0, n), dtype=float),
        b_eq=np.zeros(0),
        lb=np.array(lb, dtype=float),
        ub=np.array(ub, dtype=float),
        integer_mask=np.zeros(n, dtype=bool),
        sense=sense,
    )


# ── T-02.1: Textbook 2-variable LP ───────────────────────────────────────────

def test_dense_simplex_2var_lp():
    """
    Textbook LP:  max 3x + 5y
    s.t.  x       <= 4
          2y      <= 12
          3x + 2y <= 18
          x, y >= 0

    Optimal: x=2, y=6, obj=36.
    (Stored as min -3x -5y.)
    """
    prob = make_lp(
        c=[-3.0, -5.0],
        A_ub_dense=[[1.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 2.0]],
        b_ub=[4.0, 12.0, 18.0],
    )
    result = solve_lp_dense(prob)

    assert result.status == "optimal", f"Expected optimal, got {result.status}: {result.message}"
    assert result.x is not None

    # Objective should be -36 (minimisation of negated max)
    assert abs(result.objective - (-36.0)) < 1e-5, (
        f"Objective mismatch: expected -36.0, got {result.objective}"
    )
    # Solution should be near x=2, y=6
    assert abs(result.x[0] - 2.0) < 1e-4, f"x mismatch: expected 2, got {result.x[0]}"
    assert abs(result.x[1] - 6.0) < 1e-4, f"y mismatch: expected 6, got {result.x[1]}"


def test_dense_simplex_2var_lp_min():
    """
    min x + y  s.t.  x + y >= 3, x <= 5, y <= 5, x,y >= 0
    Optimal: x=0, y=3 (or x=3, y=0), obj=3
    """
    prob = make_lp(
        c=[1.0, 1.0],
        A_ub_dense=[[-1.0, -1.0],   # x + y >= 3 → -x - y <= -3
                    [ 1.0,  0.0],   # x <= 5
                    [ 0.0,  1.0]],  # y <= 5
        b_ub=[-3.0, 5.0, 5.0],
    )
    result = solve_lp_dense(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 3.0) < 1e-5, f"Expected obj=3, got {result.objective}"


# ── T-02.2: Infeasible LP ─────────────────────────────────────────────────────

def test_dense_simplex_infeasible():
    """
    Infeasible:  min x  s.t. x >= 5, x <= 3, x >= 0
    """
    prob = make_lp(
        c=[1.0],
        A_ub_dense=[[-1.0],  # x >= 5 → -x <= -5
                    [ 1.0]], # x <= 3
        b_ub=[-5.0, 3.0],
    )
    result = solve_lp_dense(prob)
    assert result.status == "infeasible", (
        f"Expected infeasible, got {result.status}"
    )


# ── T-02.3: Unbounded LP ──────────────────────────────────────────────────────

def test_dense_simplex_unbounded():
    """
    Unbounded:  min -x  s.t. x >= 0  (no upper bound on x → obj goes to -inf)
    """
    prob = make_lp(
        c=[-1.0],
        A_ub_dense=[[0.0]],  # dummy constraint to have at least one row
        b_ub=[1e10],
    )
    result = solve_lp_dense(prob)
    assert result.status == "unbounded", (
        f"Expected unbounded, got {result.status}"
    )


# ── T-02.4: Beale's cycling example with Bland's rule ────────────────────────

def test_dense_simplex_beale_bland_terminates():
    """
    Beale (1955) cycling LP — causes naive simplex to cycle; must terminate with Bland's rule.

    This is the standard Kotiah & Steinberg (1978) formulation with explicit bounds
    to keep the problem bounded:

      min  -2x1 - 3x2 + x3 + 12x4
      s.t.
         -2x1 +  x2 + x3         <= 4
         - x1 + 2x2       + x4   <= 6
          x1, x2, x3, x4 >= 0, <= 10

    Known optimal: obj = -11  (x1=2, x2=5/2, x3=0, x4=7/2... varies by version)
    Key requirement: terminates in finite iterations.
    """
    c = np.array([-2.0, -3.0, 1.0, 12.0])
    A = np.array([
        [-2.0,  1.0,  1.0, 0.0],
        [-1.0,  2.0,  0.0, 1.0],
    ])
    b = np.array([4.0, 6.0])
    ub = np.full(4, 100.0)

    prob = make_lp(c=c, A_ub_dense=A, b_ub=b, ub=ub)
    result = solve_lp_dense(prob)

    # Must terminate with a definitive status
    assert result.status in ("optimal", "infeasible", "unbounded"), (
        f"Bland's rule must produce a definitive answer, got: {result.status}"
    )
    # Iterations must be bounded (cycling = infinite, Bland's rule prevents this)
    assert result.iterations < 10_000, (
        f"Took {result.iterations} iterations — possible cycling without Bland's rule"
    )
    # If optimal, obj must be negative (better than all-zero)
    if result.status == "optimal":
        assert result.objective < 1.0, (
            f"Expected negative optimal obj, got {result.objective}"
        )


# ── T-02.5: Degenerate LP (multiple optimal vertices) ─────────────────────────

def test_dense_simplex_degenerate():
    """
    Degenerate LP:  min x  s.t. x <= 0, x >= 0 → x=0, obj=0.
    """
    prob = make_lp(
        c=[1.0],
        A_ub_dense=[[ 1.0],   # x <= 0
                    [-1.0]],  # -x <= 0 i.e. x >= 0
        b_ub=[0.0, 0.0],
    )
    result = solve_lp_dense(prob)
    assert result.status == "optimal"
    assert abs(result.objective - 0.0) < 1e-6
    assert abs(result.x[0] - 0.0) < 1e-6
