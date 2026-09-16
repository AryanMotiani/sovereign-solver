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
        self._result = None

    # ── Problem loading ───────────────────────────────────────────────────────

    def read_mps(self, path: str) -> "Solver":
        """Load a problem from an MPS file. Returns self for chaining."""
        self.problem = read_mps(path)
        return self

    def read_lp(self, path: str) -> "Solver":
        """Load a problem from an LP format file. Returns self for chaining."""
        from solver.io.lp_reader import read_lp as _read_lp
        self.problem = _read_lp(path)
        return self

    def read_qps(self, path: str) -> "Solver":
        """Load a QP from a QPS file (MPS + QUADOBJ section). Returns self."""
        from solver.io.qps_reader import read_qps as _read_qps
        self.problem = _read_qps(path)
        return self

    def set_problem(self, problem: Problem) -> "Solver":
        """Set the problem directly from a Problem dataclass."""
        problem.validate()
        self.problem = problem
        return self

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, method: str = "auto"):
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
            - 'milp'    : branch-and-cut (Phase 2B+, with presolve/cuts/heuristics)
            - 'qp'      : ADMM (Phase 4A)

        Returns
        -------
        SolveResult | MILPSolveResult | QPResult depending on method.
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
                method = "revised"  # revised simplex is default for LP

        if method == "simplex":
            result = solve_lp_dense(prob)

        elif method == "revised":
            from solver.lp.simplex_revised import solve_lp_revised
            result = solve_lp_revised(prob)

        elif method == "ipm":
            from solver.lp.interior_point import solve_lp_ipm
            result = solve_lp_ipm(prob)

        elif method == "pdhg":
            from solver.lp.pdhg import solve_lp_pdhg
            result = solve_lp_pdhg(prob)

        elif method == "milp":
            from solver.milp.branch_and_bound import solve_milp
            result = solve_milp(prob)

        elif method == "qp":
            from solver.qp.admm import solve_qp_admm, make_qp, QPProblem
            # Convert Problem → QPProblem if necessary
            if isinstance(prob, QPProblem):
                qp = prob
            else:
                Q = prob.P_qp if prob.P_qp is not None else None
                qp = make_qp(
                    Q=Q,
                    c=prob.c,
                    A_eq=prob.A_eq if prob.n_eq > 0 else None,
                    b_eq=prob.b_eq if prob.n_eq > 0 else None,
                    A_ub=prob.A_ub if prob.n_ineq > 0 else None,
                    b_ub=prob.b_ub if prob.n_ineq > 0 else None,
                    lb=prob.lb,
                    ub=prob.ub,
                )
            result = solve_qp_admm(qp)

        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Choose: auto, simplex, revised, ipm, pdhg, milp, qp"
            )

        self._result = result
        return result

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def result(self):
        """The result of the last solve() call."""
        return self._result

    def __repr__(self) -> str:
        prob_str = repr(self.problem) if self.problem else "no problem loaded"
        return f"Solver({prob_str})"
