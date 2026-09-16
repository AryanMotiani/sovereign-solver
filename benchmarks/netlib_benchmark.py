"""
benchmarks/netlib_benchmark.py
-------------------------------
Netlib LP benchmark: download + solve standard Netlib instances,
compare against HiGHS (if available) or scipy.optimize.linprog.

Netlib instances used (small/medium — fast to run):
  AFIRO, ADLITTLE, BLEND, SC50A, SC50B, SHARE2B, STOCFOR1

Results are written to:
  benchmarks/results/netlib_benchmark.csv

Usage:
    python -m benchmarks.netlib_benchmark
"""

from __future__ import annotations

import csv
import os
import sys
import time
import urllib.request
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
INSTANCE_CACHE = ROOT_DIR / "instances" / "netlib"

# ── Netlib instance registry ──────────────────────────────────────────────────
# (name, url, known_obj) — known_obj sourced from Netlib documentation
# MPS files sourced from HiGHS test suite (proper MPS format)
_HG = "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances"
NETLIB_INSTANCES = [
    ("afiro",    f"{_HG}/afiro.mps",        -464.7531428),
    ("adlittle", f"{_HG}/adlittle.mps",     225494.9631),
    ("lseu",     f"{_HG}/lseu.mps",         1120.0),      # MILP instance
    ("avgas",    f"{_HG}/avgas.mps",         -6.985202500e+01),
]


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_instance(name: str, url: str) -> Path:
    """Download instance to cache dir, return path. Returns cached if exists."""
    INSTANCE_CACHE.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_CACHE / f"{name}.mps"
    if path.exists():
        return path
    print(f"  Downloading {name}... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(path))
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return None
    return path


# ── Solver wrappers ───────────────────────────────────────────────────────────

def _solve_sovereign(path: Path, method: str = "simplex"):
    """Solve with our sovereign solver. Returns (status, obj, time_s, iters)."""
    from solver.io.mps_reader import read_mps
    from solver.lp.simplex_revised import solve_lp_revised
    from solver.lp.interior_point import solve_lp_ipm

    try:
        prob = read_mps(str(path))
    except Exception as e:
        return "parse_error", float("nan"), 0.0, 0

    try:
        t0 = time.perf_counter()
        if method == "simplex":
            res = solve_lp_revised(prob)
        elif method == "ipm":
            res = solve_lp_ipm(prob)
        else:
            res = solve_lp_revised(prob)
        elapsed = time.perf_counter() - t0
        return res.status, res.objective, elapsed, res.iterations
    except Exception as e:
        return "error", float("nan"), 0.0, 0


def _solve_highs(path: Path):
    """Solve with HiGHS. Returns (status, obj, time_s, iters) or None if unavailable."""
    try:
        import highspy
    except ImportError:
        return None

    try:
        h = highspy.Highs()
        h.silent()
        h.readModel(str(path))
        t0 = time.perf_counter()
        h.run()
        elapsed = time.perf_counter() - t0
        info = h.getInfoValue("primal_solution_status")[1]
        obj  = h.getInfoValue("objective_function_value")[1]
        iters = h.getInfoValue("simplex_iteration_count")[1]
        status_code = h.getModelStatus()
        status = "optimal" if "Optimal" in str(status_code) else str(status_code)
        return status, obj, elapsed, int(iters)
    except Exception as e:
        return "highs_error", float("nan"), 0.0, 0


def _solve_scipy(path: Path):
    """Solve with scipy.optimize.linprog (as reference). Returns (status, obj, time_s, iters)."""
    from scipy.optimize import linprog
    from solver.io.mps_reader import read_mps

    try:
        prob = read_mps(str(path))
    except Exception:
        return "parse_error", float("nan"), 0.0, 0

    try:
        t0 = time.perf_counter()
        A_ub = prob.A_ub.toarray() if prob.n_ineq > 0 else None
        b_ub = prob.b_ub if prob.n_ineq > 0 else None
        A_eq = prob.A_eq.toarray() if prob.n_eq > 0 else None
        b_eq = prob.b_eq if prob.n_eq > 0 else None
        bounds = list(zip(prob.lb.tolist(), [None if np.isinf(u) else u for u in prob.ub.tolist()]))
        res = linprog(prob.c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        elapsed = time.perf_counter() - t0
        status = "optimal" if res.status == 0 else ("infeasible" if res.status == 2 else "other")
        obj = float(res.fun) if res.fun is not None else float("nan")
        iters = int(getattr(res, "nit", 0))
        return status, obj, elapsed, iters
    except Exception as e:
        return "error", float("nan"), 0.0, 0


# ── Main ──────────────────────────────────────────────────────────────────────

def run_netlib_benchmark():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "netlib_benchmark.csv"

    # Check HiGHS availability
    highs_available = False
    try:
        import highspy
        highs_available = True
        print("[OK] HiGHS available -- running comparison")
    except ImportError:
        print("[!!] HiGHS not available -- using scipy.optimize.linprog as reference")

    # Check scipy availability
    scipy_ref_available = False
    try:
        from scipy.optimize import linprog
        scipy_ref_available = True
    except ImportError:
        pass

    rows = []
    header = ["instance", "n_vars", "n_ineq",
              "known_obj",
              "sovereign_simplex_status", "sovereign_simplex_obj", "sovereign_simplex_time_s", "sovereign_simplex_iters",
              "sovereign_ipm_status", "sovereign_ipm_obj", "sovereign_ipm_time_s", "sovereign_ipm_iters",
              "ref_solver", "ref_status", "ref_obj", "ref_time_s",
              "obj_match_simplex", "obj_match_ipm", "rel_err_simplex", "rel_err_ipm"]

    print("\n" + "=" * 78)
    print("   SOVEREIGN SOLVER — NETLIB BENCHMARK")
    print("=" * 78)

    for name, url, known_obj in NETLIB_INSTANCES:
        print(f"\n[{name}]  known_obj={known_obj:.4f}")
        path = _download_instance(name, url)
        if path is None:
            print(f"  SKIP: download failed")
            continue

        # Read problem dimensions
        try:
            from solver.io.mps_reader import read_mps
            prob = read_mps(str(path))
            n_vars = prob.n_vars
            n_ineq = prob.n_ineq
        except Exception:
            n_vars = n_ineq = -1

        print(f"  n_vars={n_vars}, n_ineq={n_ineq}")

        # Sovereign simplex
        ss_status, ss_obj, ss_time, ss_iters = _solve_sovereign(path, "simplex")
        print(f"  Sovereign simplex : status={ss_status}, obj={ss_obj:.6g}, t={ss_time:.3f}s, iters={ss_iters}")

        # Sovereign IPM
        si_status, si_obj, si_time, si_iters = _solve_sovereign(path, "ipm")
        print(f"  Sovereign IPM     : status={si_status}, obj={si_obj:.6g}, t={si_time:.3f}s, iters={si_iters}")

        # Reference solver
        if highs_available:
            ref_result = _solve_highs(path)
            ref_solver = "HiGHS"
        elif scipy_ref_available:
            ref_result = _solve_scipy(path)
            ref_solver = "scipy.linprog"
        else:
            ref_result = None
            ref_solver = "none"

        if ref_result is not None:
            ref_status, ref_obj, ref_time, _ = ref_result
            print(f"  {ref_solver:<18}: status={ref_status}, obj={ref_obj:.6g}, t={ref_time:.3f}s")
        else:
            ref_status, ref_obj, ref_time = "n/a", float("nan"), 0.0

        def _rel_err(our_obj, ref_o):
            if np.isnan(our_obj) or np.isnan(ref_o):
                return float("nan")
            return abs(our_obj - ref_o) / max(1.0, abs(ref_o))

        # Use known_obj or ref_obj as ground truth
        ground_truth = ref_obj if ref_result is not None and not np.isnan(ref_obj) else known_obj
        ss_err = _rel_err(ss_obj, ground_truth)
        si_err = _rel_err(si_obj, ground_truth)
        ss_match = ss_status == "optimal" and ss_err < 1e-4
        si_match = si_status == "optimal" and si_err < 1e-4

        print(f"  rel_err simplex={ss_err:.2e}  rel_err ipm={si_err:.2e}"
              f"  {'OK' if ss_match else 'XX'} simplex  {'OK' if si_match else 'XX'} ipm")

        rows.append([
            name, n_vars, n_ineq, known_obj,
            ss_status, f"{ss_obj:.8g}", f"{ss_time:.4f}", ss_iters,
            si_status, f"{si_obj:.8g}", f"{si_time:.4f}", si_iters,
            ref_solver, ref_status, f"{ref_obj:.8g}", f"{ref_time:.4f}",
            ss_match, si_match, f"{ss_err:.3e}", f"{si_err:.3e}",
        ])

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[OK] Results written to: {out_path}")

    # Summary
    simplex_wins = sum(1 for r in rows if r[16])  # obj_match_simplex
    ipm_wins     = sum(1 for r in rows if r[17])  # obj_match_ipm
    n = len(rows)
    print(f"\n  Simplex optimal match: {simplex_wins}/{n}")
    print(f"  IPM     optimal match: {ipm_wins}/{n}")


if __name__ == "__main__":
    run_netlib_benchmark()
