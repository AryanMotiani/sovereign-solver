"""
benchmarks/run_benchmarks.py
-----------------------------
Phase 5: Benchmark runner for the Sovereign Solver.

Runs all solver components on a standard suite of test problems and
produces a summary CSV and printed report.

Benchmark categories:
  A. LP Simplex (Phases 1A/1B): dense and sparse LP instances
  B. LP Interior-Point (Phase 1B): Mehrotra IPM on the same instances
  C. LP PDHG (Phase 1C): first-order method benchmarks
  D. MILP B&B (Phase 2B): binary and mixed-integer problems
  E. QP ADMM (Phase 4A): quadratic programs
  F. Domain demos (Phase 4B): crude blend and refinery scheduling

Each benchmark row in the CSV records:
  solver, problem_name, n_vars, n_ineq, objective, time_s, iters, status

Comparison table:
  For LP instances, show Simplex vs IPM vs PDHG side-by-side.

Usage:
    python -m benchmarks.run_benchmarks
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from solver.lp.simplex_revised import solve_lp_revised
from solver.lp.interior_point import solve_lp_ipm
from solver.lp.pdhg import solve_lp_pdhg
from solver.milp.branch_and_bound import solve_milp
from solver.qp.admm import make_qp, solve_qp_admm
from solver.problem import Problem

# ── Result data container ─────────────────────────────────────────────────────

@dataclass
class BenchRow:
    solver:   str
    problem:  str
    n_vars:   int
    n_ineq:   int
    status:   str
    obj:      float
    time_s:   float
    iters:    int


# ── Problem generators ────────────────────────────────────────────────────────

def lp_problem(c, A_ub, b_ub, name="lp") -> Problem:
    n = len(c)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.zeros(n),
        ub=np.full(n, np.inf),
        integer_mask=np.zeros(n, dtype=bool),
        name=name,
    )


def milp_problem(c, A_ub, b_ub, integer_vars, ub_vars=None, name="milp") -> Problem:
    n = len(c)
    ub = np.full(n, np.inf) if ub_vars is None else np.array(ub_vars, dtype=float)
    int_mask = np.zeros(n, dtype=bool)
    for j in integer_vars:
        int_mask[j] = True
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.zeros(n),
        ub=ub,
        integer_mask=int_mask,
        name=name,
    )


def random_lp(n: int, m: int, seed: int = 0) -> Problem:
    """Random dense LP with guaranteed bounded feasible solution."""
    rng = np.random.default_rng(seed)
    A = np.abs(rng.standard_normal((m, n)))
    x_true = rng.uniform(0.5, 2.0, n)
    b = A @ x_true + rng.uniform(0.1, 1.0, m)
    # Add upper bound rows to keep bounded
    A_ub = np.vstack([A, np.eye(n)])
    b_ub = np.concatenate([b, np.full(n, 5.0)])
    c = rng.standard_normal(n)
    return lp_problem(c, A_ub, b_ub, name=f"random_lp_n{n}_m{m}")


def random_knapsack(n: int, seed: int = 0) -> Problem:
    """Random binary 0-1 knapsack."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(1, 5, n)
    values  = rng.uniform(1, 10, n)
    cap = weights.sum() * 0.6
    return milp_problem(
        c=(-values).tolist(), A_ub=[weights.tolist()], b_ub=[cap],
        integer_vars=list(range(n)), ub_vars=[1.0]*n,
        name=f"knapsack_n{n}",
    )


# ── Benchmark runner ──────────────────────────────────────────────────────────

def bench_lp(prob: Problem, rows: list):
    """Run Simplex, IPM, and PDHG on an LP; append rows."""
    for solver_name, fn in [
        ("simplex", solve_lp_revised),
        ("ipm",     solve_lp_ipm),
        ("pdhg",    solve_lp_pdhg),
    ]:
        t0 = time.monotonic()
        res = fn(prob)
        elapsed = time.monotonic() - t0
        iters = getattr(res, "iterations", None) or getattr(res, "iters", 0) or 0
        rows.append(BenchRow(
            solver=solver_name,
            problem=prob.name,
            n_vars=prob.n_vars,
            n_ineq=prob.n_ineq,
            status=res.status,
            obj=res.objective if res.objective is not None else float("inf"),
            time_s=elapsed,
            iters=int(iters),
        ))


def bench_milp(prob: Problem, rows: list):
    """Run B&B on a MILP; append row."""
    t0 = time.monotonic()
    res = solve_milp(prob, time_limit=30.0)
    elapsed = time.monotonic() - t0
    rows.append(BenchRow(
        solver="branch_and_bound",
        problem=prob.name,
        n_vars=prob.n_vars,
        n_ineq=prob.n_ineq,
        status=res.status,
        obj=res.objective,
        time_s=elapsed,
        iters=res.nodes,
    ))


def bench_qp(name, Q, c, A_ub=None, b_ub=None, rows: list = None):
    """Run ADMM on a QP; append row."""
    prob = make_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub)
    t0 = time.monotonic()
    res = solve_qp_admm(prob)
    elapsed = time.monotonic() - t0
    rows.append(BenchRow(
        solver="admm_qp",
        problem=name,
        n_vars=prob.n,
        n_ineq=prob.m_ub,
        status=res.status,
        obj=res.objective,
        time_s=elapsed,
        iters=res.iters,
    ))


# ── Benchmark suite ───────────────────────────────────────────────────────────

def run_all() -> list:
    rows: list[BenchRow] = []

    print("\n" + "=" * 70)
    print("  SOVEREIGN SOLVER — BENCHMARK SUITE (Phase 5)")
    print("=" * 70)

    # ── A. LP benchmarks ──────────────────────────────────────────────────────
    print("\n[A] LP benchmarks (Simplex / IPM / PDHG)")
    lp_problems = [
        lp_problem([-3,-5], [[1,0],[0,2],[3,2]], [4,12,18], "2var_lp"),
        lp_problem([-1,-2,-3], [[1,1,0],[0,1,1],[1,0,1]], [4,4,4], "3var_lp"),
        random_lp(n=10,  m=6,  seed=1),
        random_lp(n=20,  m=12, seed=2),
        random_lp(n=50,  m=30, seed=3),
    ]
    for prob in lp_problems:
        bench_lp(prob, rows)
        print(f"  {prob.name}: done")

    # ── D. MILP benchmarks ────────────────────────────────────────────────────
    print("\n[D] MILP benchmarks (Branch-and-Bound)")
    milp_problems = [
        milp_problem([-3,-5], [[1,0],[0,2],[3,2]], [4,12,18], [0,1], name="2var_milp"),
        random_knapsack(n=8,  seed=10),
        random_knapsack(n=12, seed=11),
        random_knapsack(n=16, seed=12),
    ]
    for prob in milp_problems:
        bench_milp(prob, rows)
        print(f"  {prob.name}: done")

    # ── E. QP benchmarks ──────────────────────────────────────────────────────
    print("\n[E] QP benchmarks (ADMM)")
    rng = np.random.default_rng(99)
    for n_qp in [5, 10, 20]:
        M = rng.standard_normal((n_qp, n_qp))
        Q = M.T @ M + np.eye(n_qp) * 0.1
        c = rng.standard_normal(n_qp)
        bench_qp(f"qp_n{n_qp}", Q=Q, c=c, rows=rows)
        print(f"  qp_n{n_qp}: done")

    # ── F. Domain demos ───────────────────────────────────────────────────────
    print("\n[F] Domain demos (crude blend + refinery scheduling)")
    # Crude blend LP
    from demos.crude_blend_lp import build_blend_problem
    blend_prob = build_blend_problem()
    bench_lp(blend_prob, rows)
    print("  crude_blend_lp: done")

    # Refinery scheduling MILP
    from demos.refinery_scheduling_milp import build_schedule_problem
    sched_prob = build_schedule_problem()
    bench_milp(sched_prob, rows)
    print("  refinery_schedule_milp: done")

    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(rows: list):
    print("\n" + "=" * 90)
    print(f"{'Solver':<18} {'Problem':<28} {'n':>5} {'m':>5} {'Status':<12} {'Obj':>12} {'Time(s)':>9} {'Iters':>8}")
    print("-" * 90)
    for r in rows:
        obj_str = f"{r.obj:.4g}" if np.isfinite(r.obj) else "inf"
        print(f"{r.solver:<18} {r.problem:<28} {r.n_vars:>5} {r.n_ineq:>5} {r.status:<12} {obj_str:>12} {r.time_s:>9.4f} {r.iters:>8}")
    print("=" * 90)


def save_csv_results(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "problem", "n_vars", "n_ineq", "status", "obj", "time_s", "iters"])
        for r in rows:
            writer.writerow([r.solver, r.problem, r.n_vars, r.n_ineq, r.status,
                             f"{r.obj:.6g}", f"{r.time_s:.6f}", r.iters])


def print_lp_comparison(rows: list):
    """Print a side-by-side LP solver comparison table."""
    lp_rows = [r for r in rows if r.solver in ("simplex", "ipm", "pdhg")]
    problems = list(dict.fromkeys(r.problem for r in lp_rows))

    print("\n+" + "-" * 67 + "+")
    print("|       LP SOLVER COMPARISON: Simplex vs IPM vs PDHG              |")
    print("+" + "-" * 30 + "+" + "-" * 36 + "+")
    print(f"| {'Problem':<29}| {'Simplex':>10} {'IPM':>10} {'PDHG':>10}   |")
    print("+" + "-" * 30 + "+" + "-" * 36 + "+")
    for prob in problems:
        times = {}
        objs  = {}
        for r in lp_rows:
            if r.problem == prob:
                times[r.solver] = r.time_s
                objs[r.solver]  = r.obj
        t_s = times.get("simplex", float("nan"))
        t_i = times.get("ipm",     float("nan"))
        t_p = times.get("pdhg",    float("nan"))
        print(f"| {prob:<29}| {t_s:>9.4f}s {t_i:>9.4f}s {t_p:>9.4f}s |")
    print("+" + "-" * 30 + "+" + "-" * 36 + "+")


if __name__ == "__main__":
    rows = run_all()
    print_table(rows)
    print_lp_comparison(rows)

    out_csv = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.csv")
    save_csv_results(rows, os.path.abspath(out_csv))
    print(f"\nResults saved to: benchmarks/results/benchmark_results.csv")
    print(f"Total benchmarks run: {len(rows)}")
