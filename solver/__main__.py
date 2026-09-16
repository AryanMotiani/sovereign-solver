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

import numpy as np

from solver.solver import Solver
from solver.io.mps_reader import read_mps


def cmd_solve(args: argparse.Namespace) -> None:
    s = Solver()
    print(f"Reading: {args.file}")
    try:
        fpath = args.file.lower()
        if fpath.endswith(".lp"):
            from solver.io.lp_reader import read_lp
            prob = read_lp(args.file)
            s.set_problem(prob)
        elif fpath.endswith(".qps"):
            from solver.io.qps_reader import read_qps
            prob = read_qps(args.file)
            s.set_problem(prob)
        else:
            s.read_mps(args.file)
    except Exception as e:
        print(f"ERROR reading file: {e}", file=sys.stderr)
        sys.exit(1)

    prob = s.problem
    print(f"Problem: {prob}")
    print(f"Method:  {args.method}")
    if hasattr(args, "device") and args.device:
        print(f"Device:  {args.device}")
    print("Solving ...")

    t0 = time.perf_counter()
    try:
        kwargs = {"method": args.method}
        if hasattr(args, "device") and args.device:
            kwargs["device"] = args.device
        result = s.solve(**kwargs)
    except Exception as e:
        print(f"ERROR during solve: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Result ---")
    print(f"Status:     {result.status}")
    print(f"Objective:  {result.objective:.10g}")
    # MILPSolveResult uses .nodes; LP uses .iterations
    if hasattr(result, "nodes"):
        print(f"Nodes:      {result.nodes}")
        if hasattr(result, "lp_relaxation"):
            print(f"LP bound:   {result.lp_relaxation:.10g}")
        if hasattr(result, "gap"):
            print(f"Gap:        {result.gap*100:.4f}%")
    elif hasattr(result, "iterations"):
        print(f"Iterations: {result.iterations}")
    print(f"Wall clock: {elapsed:.3f}s")
    if hasattr(result, "message") and result.message:
        print(f"Message:    {result.message}")
    if result.x is not None and prob.n_vars <= 20:
        print(f"Solution:   {result.x}")
    elif result.x is not None and args.verbose:
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print(f"Solution ({prob.n_vars} vars):")
        print(result.x)


def cmd_info(args: argparse.Namespace) -> None:
    try:
        fpath = args.file.lower()
        if fpath.endswith(".lp"):
            from solver.io.lp_reader import read_lp
            prob = read_lp(args.file)
        else:
            prob = read_mps(args.file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(prob)
    print(f"  Variables:    {prob.n_vars}")
    print(f"  Inequalities: {prob.n_ineq}")
    print(f"  Equalities:   {prob.n_eq}")
    print(f"  Integer vars: {prob.integer_mask.sum()}")
    print(f"  Type:         {'MILP' if prob.is_milp else ('QP' if prob.is_qp else 'LP')}")
    print(f"  Sense:        {prob.sense}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m solver",
        description="Sovereign Optimization Solver — CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve command
    p_solve = subparsers.add_parser("solve", help="Solve an MPS or LP file")
    p_solve.add_argument("file", help="Path to .mps or .lp file")
    p_solve.add_argument(
        "--method",
        default="auto",
        choices=["auto", "simplex", "revised", "ipm", "pdhg", "pdhg_gpu", "milp", "qp"],
        help="Solver method (default: auto)",
    )
    p_solve.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for GPU methods (default: auto)",
    )
    p_solve.add_argument("--verbose", action="store_true", help="Print full solution vector")

    # info command
    p_info = subparsers.add_parser("info", help="Show problem dimensions")
    p_info.add_argument("file", help="Path to .mps or .lp file")

    args = parser.parse_args()
    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
