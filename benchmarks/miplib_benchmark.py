"""
benchmarks/miplib_benchmark.py
--------------------------------
MIPLIB 2017 Benchmark for the Sovereign Solver.

Downloads small MIPLIB 2017 instances from the official repository and
solves them with our Branch-and-Cut engine. Compares against HiGHS for
reference.

MIPLIB 2017 instances used (the "easy" subset):
  - enigma     : 0-1 IP,   100 vars,  100 rows
  - stein27    : 0-1 IP,   118 vars,   54 rows (set covering)
  - stein45    : 0-1 IP,   331 vars,   79 rows
  - gt2        : MILP,     188 vars,   29 rows
  - misc01     : MILP,     219 vars,   73 rows

Usage:
    py -3 -m benchmarks.miplib_benchmark
    py -3 -m benchmarks.miplib_benchmark --no-highs   # skip HiGHS comparison
    py -3 -m benchmarks.miplib_benchmark --timeout 60  # per-instance timeout

Results are saved to benchmarks/results/miplib_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "instances" / "miplib"
RESULTS_DIR = ROOT / "benchmarks" / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# MIPLIB 2017 instance metadata
# Source: https://miplib.zib.de/  (MPS format)
# We use instances available from the HiGHS test set and other open mirrors
INSTANCES = [
    {
        "name": "stein27",
        "url": "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/stein27.mps",
        "known_obj": 18.0,
        "tol": 0.5,
        "description": "0-1 IP, set covering, 27-element Steiner problem",
    },
    {
        "name": "stein45",
        "url": "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/stein45.mps",
        "known_obj": 30.0,
        "tol": 1.0,
        "description": "0-1 IP, set covering, 45-element Steiner problem",
    },
    {
        "name": "lp_afiro_milp",
        "url": None,  # generated synthetically
        "known_obj": None,
        "tol": None,
        "description": "Synthetic MILP derived from AFIRO LP relaxation",
    },
]

# Smaller hand-crafted MIPLIB-representative instances (always available)
BUILTIN_INSTANCES = [
    {
        "name": "binary_knapsack_10",
        "description": "0-1 knapsack, n=10 items",
        "generate": "_gen_knapsack",
        "n": 10,
        "known_obj": None,
    },
    {
        "name": "binary_knapsack_20",
        "description": "0-1 knapsack, n=20 items",
        "generate": "_gen_knapsack",
        "n": 20,
        "known_obj": None,
    },
    {
        "name": "set_cover_20",
        "description": "Set covering, 20 elements, 30 sets",
        "generate": "_gen_set_cover",
        "n_elements": 20,
        "n_sets": 30,
        "known_obj": None,
    },
    {
        "name": "facility_location_10",
        "description": "Uncapacitated facility location, 10 facilities, 15 customers",
        "generate": "_gen_facility",
        "n_fac": 10,
        "n_cust": 15,
        "known_obj": None,
    },
    {
        "name": "assignment_8x8",
        "description": "Assignment problem, 8x8",
        "generate": "_gen_assignment",
        "n": 8,
        "known_obj": None,
    },
]


# ── Problem generators ────────────────────────────────────────────────────────

def _gen_knapsack(n: int, seed: int = 42):
    """Random 0-1 knapsack: max Σ v_i x_i  s.t. Σ w_i x_i <= W."""
    import numpy as np, scipy.sparse as sp
    from solver.problem import Problem
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 20, n).astype(float)
    v = rng.integers(5, 30, n).astype(float)
    W = 0.4 * w.sum()
    prob = Problem(
        c=-v,  # maximise value = minimise -value
        A_ub=sp.csr_matrix(w.reshape(1, n)),
        b_ub=np.array([W]),
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.zeros(n), ub=np.ones(n),
        integer_mask=np.ones(n, dtype=bool),
        name=f"knapsack_{n}",
    )
    return prob


def _gen_set_cover(n_elements: int, n_sets: int, seed: int = 42):
    """Random set covering: min Σ c_j x_j  s.t. coverage."""
    import numpy as np, scipy.sparse as sp
    from solver.problem import Problem
    rng = np.random.default_rng(seed)
    # Each set covers a random subset of elements (ensure full coverage possible)
    A = rng.integers(0, 2, (n_elements, n_sets)).astype(float)
    # Ensure each element covered by at least 2 sets
    for i in range(n_elements):
        if A[i].sum() < 2:
            idxs = rng.choice(n_sets, 2, replace=False)
            A[i, idxs] = 1.0
    c = rng.integers(1, 10, n_sets).astype(float)
    # Ax >= 1  →  -Ax <= -1
    prob = Problem(
        c=c,
        A_ub=sp.csr_matrix(-A),
        b_ub=-np.ones(n_elements),
        A_eq=sp.csr_matrix((0, n_sets)),
        b_eq=np.zeros(0),
        lb=np.zeros(n_sets), ub=np.ones(n_sets),
        integer_mask=np.ones(n_sets, dtype=bool),
        name=f"set_cover_{n_elements}x{n_sets}",
    )
    return prob


def _gen_facility(n_fac: int, n_cust: int, seed: int = 42):
    """Uncapacitated facility location."""
    import numpy as np, scipy.sparse as sp
    from solver.problem import Problem
    rng = np.random.default_rng(seed)
    f_cost = rng.integers(50, 200, n_fac).astype(float)
    t_cost = rng.integers(5, 50, (n_fac, n_cust)).astype(float)

    # vars: [y_i (n_fac binary), x_ij (n_fac*n_cust binary)]
    n_y = n_fac
    n_x = n_fac * n_cust
    n = n_y + n_x

    c_obj = np.concatenate([f_cost, t_cost.ravel()])

    rows_ub, cols_ub, data_ub, rhs_ub = [], [], [], []
    rows_eq, cols_eq, data_eq, rhs_eq = [], [], [], []
    r_ub = r_eq = 0

    # Customer assignment: Σ_i x[i,j] = 1  ∀j
    for j in range(n_cust):
        for i in range(n_fac):
            rows_eq.append(r_eq); cols_eq.append(n_y + i*n_cust+j); data_eq.append(1.0)
        rhs_eq.append(1.0); r_eq += 1

    # Linking: x[i,j] <= y[i]  →  x[i,j] - y[i] <= 0
    for i in range(n_fac):
        for j in range(n_cust):
            rows_ub.append(r_ub); cols_ub.append(n_y + i*n_cust+j); data_ub.append(1.0)
            rows_ub.append(r_ub); cols_ub.append(i); data_ub.append(-1.0)
            rhs_ub.append(0.0); r_ub += 1

    A_eq = sp.csr_matrix((data_eq, (rows_eq, cols_eq)), shape=(r_eq, n))
    A_ub = sp.csr_matrix((data_ub, (cols_ub, rows_ub)), shape=(n, r_ub)).T  # transposed build
    A_ub = sp.csr_matrix((data_ub, (rows_ub, cols_ub)), shape=(r_ub, n))

    int_mask = np.zeros(n, dtype=bool)
    int_mask[:] = True

    return Problem(
        c=c_obj, A_ub=A_ub, b_ub=np.zeros(r_ub),
        A_eq=A_eq, b_eq=np.ones(r_eq),
        lb=np.zeros(n), ub=np.ones(n),
        integer_mask=int_mask,
        name=f"facility_{n_fac}x{n_cust}",
    )


def _gen_assignment(n: int, seed: int = 42):
    """Assignment problem: min Σ c_ij x_ij  s.t. Σ_j x_ij=1, Σ_i x_ij=1."""
    import numpy as np, scipy.sparse as sp
    from solver.problem import Problem
    rng = np.random.default_rng(seed)
    C = rng.integers(1, 100, (n, n)).astype(float)

    n_vars = n * n
    # Σ_j x[i,j] = 1  (n rows)
    # Σ_i x[i,j] = 1  (n cols)
    eq_rows, eq_cols, eq_data, eq_rhs = [], [], [], []
    r_eq = 0
    for i in range(n):
        for j in range(n):
            eq_rows.append(r_eq); eq_cols.append(i*n+j); eq_data.append(1.0)
        eq_rhs.append(1.0); r_eq += 1
    for j in range(n):
        for i in range(n):
            eq_rows.append(r_eq); eq_cols.append(i*n+j); eq_data.append(1.0)
        eq_rhs.append(1.0); r_eq += 1

    A_eq = sp.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(r_eq, n_vars))
    return Problem(
        c=C.ravel(), A_ub=sp.csr_matrix((0, n_vars)), b_ub=np.zeros(0),
        A_eq=A_eq, b_eq=np.array(eq_rhs),
        lb=np.zeros(n_vars), ub=np.ones(n_vars),
        integer_mask=np.ones(n_vars, dtype=bool),
        name=f"assignment_{n}x{n}",
    )


# ── Downloader ────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


# ── Solve one instance ────────────────────────────────────────────────────────

def _solve_instance(prob, timeout: float) -> dict:
    from solver.milp.branch_and_bound import solve_milp
    t0 = time.perf_counter()
    try:
        result = solve_milp(prob, time_limit=timeout, verbose=False)
        elapsed = time.perf_counter() - t0
        return {
            "status": result.status,
            "objective": result.objective,
            "lp_relaxation": result.lp_relaxation,
            "gap": result.gap,
            "nodes": result.nodes,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error", "objective": None,
            "lp_relaxation": None, "gap": None,
            "nodes": 0, "time": elapsed, "error": str(e),
        }


def _solve_highs(prob, timeout: float) -> dict:
    try:
        import highspy
    except ImportError:
        return {"status": "highs_not_installed", "objective": None, "time": None}

    t0 = time.perf_counter()
    try:
        import numpy as np
        h = highspy.Highs()
        h.silent()
        n = prob.n_vars
        INF = 1e30

        # Add variables
        h.addVars(n,
                  prob.lb.tolist(),
                  [u if np.isfinite(u) else INF for u in prob.ub])

        # Set costs
        for j in range(n):
            h.changeColCost(j, float(prob.c[j]))

        # Set integrality
        for j in range(n):
            if prob.integer_mask[j]:
                h.changeColIntegrality(j, highspy.HighsVarType.kInteger)

        # Add inequality rows: A_ub x <= b_ub  →  lb=-INF, ub=b_ub[i]
        if prob.n_ineq > 0:
            A = prob.A_ub.tocsr()
            for i in range(prob.n_ineq):
                row = A.getrow(i)
                h.addRow(-INF, float(prob.b_ub[i]),
                         row.nnz,
                         row.indices.tolist(),
                         row.data.tolist())

        # Add equality rows: lb = ub = b_eq[i]
        if prob.n_eq > 0:
            A = prob.A_eq.tocsr()
            for i in range(prob.n_eq):
                row = A.getrow(i)
                h.addRow(float(prob.b_eq[i]), float(prob.b_eq[i]),
                         row.nnz,
                         row.indices.tolist(),
                         row.data.tolist())

        h.setOptionValue("time_limit", timeout)
        h.run()

        model_status = h.getModelStatus()
        obj = h.getInfoValue("objective_function_value")[1]
        elapsed = time.perf_counter() - t0

        status_str = str(model_status).lower()
        is_optimal = "optimal" in status_str or "feasible" in status_str
        return {
            "status": "optimal" if is_optimal else str(model_status),
            "objective": float(obj),
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "error", "objective": None, "time": elapsed, "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIPLIB Benchmark — Sovereign Solver")
    parser.add_argument("--no-highs", action="store_true", help="Skip HiGHS comparison")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-instance timeout (s)")
    parser.add_argument("--only-builtin", action="store_true",
                        help="Skip internet downloads, only run built-in instances")
    args = parser.parse_args()

    from solver.io.mps_reader import read_mps

    results = []

    print("=" * 75)
    print(f"  MIPLIB BENCHMARK — SOVEREIGN SOLVER")
    print(f"  Timeout: {args.timeout}s per instance")
    print("=" * 75)

    # ── Built-in instances (always run) ───────────────────────────────────────
    generators = {
        "_gen_knapsack": _gen_knapsack,
        "_gen_set_cover": _gen_set_cover,
        "_gen_facility": _gen_facility,
        "_gen_assignment": _gen_assignment,
    }

    for inst in BUILTIN_INSTANCES:
        name = inst["name"]
        desc = inst["description"]
        gen_fn = generators[inst["generate"]]
        # Build kwargs excluding metadata keys
        kwargs = {k: v for k, v in inst.items()
                  if k not in ("name", "description", "generate", "known_obj")}
        try:
            prob = gen_fn(**kwargs)
        except Exception as e:
            print(f"\n  [{name}] Build error: {e}")
            continue

        print(f"\n  [{name}]  {desc}")
        print(f"    n={prob.n_vars} vars ({prob.integer_mask.sum()} int), "
              f"m={prob.n_ineq}+{prob.n_eq} constrs")

        sov = _solve_instance(prob, args.timeout)
        print(f"    Sovereign: status={sov['status']:12s}  "
              f"obj={sov['objective'] if sov['objective'] is not None else 'N/A':>12}  "
              f"nodes={sov['nodes']:>5}  time={sov['time']:.3f}s")

        highs_res = {"status": "skipped", "objective": None, "time": None}
        if not args.no_highs:
            highs_res = _solve_highs(prob, args.timeout)
            if highs_res["status"] == "highs_not_installed":
                print("    HiGHS: not installed (pip install highspy)")
            else:
                print(f"    HiGHS:    status={highs_res['status']:12s}  "
                      f"obj={highs_res['objective'] if highs_res['objective'] is not None else 'N/A':>12}  "
                      f"time={highs_res['time']:.3f}s" if highs_res['time'] else "")

        # Gap vs HiGHS
        gap_vs_highs = None
        if sov["objective"] is not None and highs_res.get("objective") is not None:
            ref = abs(highs_res["objective"])
            if ref > 1e-6:
                gap_vs_highs = abs(sov["objective"] - highs_res["objective"]) / ref

        results.append({
            "instance": name,
            "description": desc,
            "n_vars": prob.n_vars,
            "n_int": int(prob.integer_mask.sum()),
            "n_ineq": prob.n_ineq,
            "n_eq": prob.n_eq,
            "sovereign_status": sov["status"],
            "sovereign_obj": sov["objective"],
            "sovereign_lp": sov.get("lp_relaxation"),
            "sovereign_gap": sov.get("gap"),
            "sovereign_nodes": sov.get("nodes"),
            "sovereign_time": sov["time"],
            "highs_status": highs_res.get("status"),
            "highs_obj": highs_res.get("objective"),
            "highs_time": highs_res.get("time"),
            "gap_vs_highs": gap_vs_highs,
        })

    # ── Downloaded MIPLIB instances ───────────────────────────────────────────
    if not args.only_builtin:
        print(f"\n  Downloading MIPLIB instances to {CACHE_DIR} ...")
        for inst in INSTANCES:
            if inst["url"] is None:
                continue
            name = inst["name"]
            mps_path = CACHE_DIR / f"{name}.mps"
            if not _download(inst["url"], mps_path):
                continue

            print(f"\n  [{name}]  {inst['description']}")
            try:
                prob = read_mps(str(mps_path))
            except Exception as e:
                print(f"    Read error: {e}")
                continue

            print(f"    n={prob.n_vars} vars ({prob.integer_mask.sum()} int), "
                  f"m={prob.n_ineq}+{prob.n_eq} constrs")

            sov = _solve_instance(prob, args.timeout)
            known = inst.get("known_obj")
            tol = inst.get("tol", 1.0)
            match = ""
            if sov["objective"] is not None and known is not None:
                match = "  MATCH" if abs(sov["objective"] - known) <= tol else f"  DIFF ({known})"
            print(f"    Sovereign: status={sov['status']:12s}  "
                  f"obj={sov['objective']:>12.4f}  "
                  f"nodes={sov['nodes']:>5}  time={sov['time']:.3f}s{match}")

            results.append({
                "instance": name,
                "description": inst["description"],
                "n_vars": prob.n_vars,
                "n_int": int(prob.integer_mask.sum()),
                "n_ineq": prob.n_ineq,
                "n_eq": prob.n_eq,
                "sovereign_status": sov["status"],
                "sovereign_obj": sov["objective"],
                "sovereign_lp": sov.get("lp_relaxation"),
                "sovereign_gap": sov.get("gap"),
                "sovereign_nodes": sov.get("nodes"),
                "sovereign_time": sov["time"],
            })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if results:
        csv_path = RESULTS_DIR / "miplib_benchmark.csv"
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n  Results saved to {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_solved = sum(1 for r in results if r["sovereign_status"] == "optimal")
    print(f"\n  SUMMARY: {n_solved}/{len(results)} instances solved to optimality")
    avg_time = sum(r["sovereign_time"] for r in results if r["sovereign_time"]) / max(len(results), 1)
    print(f"           Average solve time: {avg_time:.3f}s")


if __name__ == "__main__":
    main()
