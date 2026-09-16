"""
Sovereign Optimization Solver
==============================
A sovereign mathematical optimization engine for LP, MILP, and QP.
Built from scratch — no external solver dependencies in this package.

Usage
-----
Python API::

    from solver import Solver
    s = Solver()
    s.read_mps("instance.mps")
    result = s.solve()
    print(result)

CLI::

    python -m solver solve instance.mps [--method simplex|ipm|pdhg]
"""

from solver.solver import Solver
from solver.problem import Problem
from solver.lp.simplex_dense import SolveResult
from solver.milp.branch_and_bound import MILPSolveResult
from solver.qp.admm import QPResult
from solver.io.mps_reader import read_mps
from solver.io.lp_reader import read_lp
from solver.io.qps_reader import read_qps

__version__ = "0.3.0-alpha"
__all__ = [
    "Solver", "Problem",
    "SolveResult", "MILPSolveResult", "QPResult",
    "read_mps", "read_lp", "read_qps",
]
