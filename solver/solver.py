"""
solver/solver.py
-----------------
High-level Solver class — public Python API.

Example
-------
    from solver import Solver
    s = Solver()
    s.read_mps("instance.mps")
    result = s.solve(method="simplex")
    print(result)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from solver.problem import Problem
from solver.io.mps_reader import read_mps
from solver.lp.simplex_dense import solve_lp_dense, SolveResult


class Solver:
    """
    High-level interface to the Sovereign Optimization Solver.

    Attributes
    ----------
    problem : Problem or None
        The currently loaded optimization problem.
    """

    def __init__(self) -> None:
        self.problem: Optional[Problem] = None
        self._result: Optional[SolveResult] = None

    # ── Problem loading ───────────────────────────────────────────────────────

    def read_mps(self, path: str) -> "Solver":
        """Load a problem from an MPS file. Returns self for chaining."""
        self.problem = read_mps(path)
        return self

    def set_problem(self, problem: Problem) -> "Solver":
        """Set the problem directly from a Problem dataclass."""
        problem.validate()
        self.problem = problem
        return self

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, method: str = "auto") -> SolveResult:
        """
        Solve the currently loaded problem.

        Parameters
        ----------
        method : str
            Solver method to use:
            - 'auto'    : choose based on problem type
            - 'simplex' : dense simplex (Phase 1A scaffold)
            - 'revised' : revised sparse simplex (Phase 1B)
            - 'ipm'     : Mehrotra interior-point (Phase 1B)
            - 'pdhg'    : first-order PDHG (Phase 1C)
            - 'milp'    : branch-and-cut (Phase 2B+)
            - 'qp'      : ADMM (Phase 4A)

        Returns
        -------
        SolveResult
        """
        if self.problem is None:
            raise RuntimeError("No problem loaded. Call read_mps() or set_problem() first.")

        prob = self.problem

        if method == "auto":
            if prob.is_milp:
                method = "milp"
            elif prob.is_qp:
                method = "qp"
            else:
                method = "simplex"  # will upgrade to 'revised' in Phase 1B

        if method == "simplex":
            result = solve_lp_dense(prob)

        elif method == "revised":
            try:
                from solver.lp.simplex_revised import solve_lp_revised
                result = solve_lp_revised(prob)
            except ImportError:
                raise RuntimeError("Revised simplex not yet implemented (Phase 1B).")

        elif method == "ipm":
            try:
                from solver.lp.interior_point import solve_lp_ipm
                result = solve_lp_ipm(prob)
            except ImportError:
                raise RuntimeError("IPM not yet implemented (Phase 1B).")

        elif method == "pdhg":
            try:
                from solver.lp.pdhg import solve_lp_pdhg
                result = solve_lp_pdhg(prob)
            except ImportError:
                raise RuntimeError("PDHG not yet implemented (Phase 1C).")

        elif method == "milp":
            try:
                from solver.milp.branch_and_bound import solve_milp
                result = solve_milp(prob)
            except ImportError:
                raise RuntimeError("MILP B&B not yet implemented (Phase 2B).")

        elif method == "qp":
            try:
                from solver.qp.admm import solve_qp_admm
                result = solve_qp_admm(prob)
            except ImportError:
                raise RuntimeError("QP ADMM not yet implemented (Phase 4A).")

        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Choose: auto, simplex, revised, ipm, pdhg, milp, qp"
            )

        self._result = result
        return result

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def result(self) -> Optional[SolveResult]:
        """The result of the last solve() call."""
        return self._result

    def __repr__(self) -> str:
        prob_str = repr(self.problem) if self.problem else "no problem loaded"
        return f"Solver({prob_str})"
