"""
solver/__main__.py
------------------
CLI entry point: python -m solver <command> [args]

Commands
--------
  solve <file.mps> [--method METHOD]   Solve an MPS file and print result
  info  <file.mps>                     Print problem dimensions without solving

Example
-------
  python -m solver solve instances/netlib/afiro.mps
  python -m solver solve instances/netlib/afiro.mps --method simplex
  python -m solver info  instances/miplib/stein27.mps
"""

from __future__ import annotations

import argparse
import sys
import time

from solver.solver import Solver
from solver.io.mps_reader import read_mps


def cmd_solve(args: argparse.Namespace) -> None:
    s = Solver()
    print(f"Reading: {args.file}")
    try:
        s.read_mps(args.file)
    except Exception as e:
        print(f"ERROR reading MPS: {e}", file=sys.stderr)
        sys.exit(1)

    prob = s.problem
    print(f"Problem: {prob}")
    print(f"Method:  {args.method}")
    print("Solving ...")

    t0 = time.perf_counter()
    try:
        result = s.solve(method=args.method)
    except Exception as e:
        print(f"ERROR during solve: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Result ---")
    print(f"Status:     {result.status}")
    print(f"Objective:  {result.objective:.10g}")
    print(f"Iterations: {result.iterations}")
    print(f"Wall clock: {elapsed:.3f}s")
    if result.message:
        print(f"Message:    {result.message}")
    if result.x is not None and prob.n_vars <= 20:
        print(f"Solution:   {result.x}")


def cmd_info(args: argparse.Namespace) -> None:
    try:
        prob = read_mps(args.file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(prob)
    print(f"  Variables:   {prob.n_vars}")
    print(f"  Inequalities:{prob.n_ineq}")
    print(f"  Equalities:  {prob.n_eq}")
    print(f"  Integer vars:{prob.integer_mask.sum()}")
    print(f"  Type:        {'MILP' if prob.is_milp else ('QP' if prob.is_qp else 'LP')}")
    print(f"  Sense:       {prob.sense}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m solver",
        description="Sovereign Optimization Solver — CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve command
    p_solve = subparsers.add_parser("solve", help="Solve an MPS file")
    p_solve.add_argument("file", help="Path to .mps file")
    p_solve.add_argument(
        "--method",
        default="auto",
        choices=["auto", "simplex", "revised", "ipm", "pdhg", "milp", "qp"],
        help="Solver method (default: auto)",
    )

    # info command
    p_info = subparsers.add_parser("info", help="Show problem dimensions")
    p_info.add_argument("file", help="Path to .mps file")

    args = parser.parse_args()
    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
