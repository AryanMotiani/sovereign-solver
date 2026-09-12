"""
demos/refinery_scheduling_milp.py
-----------------------------------
Demo 2: Refinery Unit Scheduling MILP (Weekly Planning)

Problem:
  A medium-scale Indian refinery must schedule 4 process units over a
  5-day horizon.  Each unit (CDU, VDU, FCC, HDS) can run in different
  modes each day, some of which require a binary on/off decision.

  Modes:
    CDU: crude distillation — capacity varies by feed quality
    VDU: vacuum distillation — continuous, dependent on CDU throughput
    FCC: fluid catalytic cracking — on/off per day (integer decision)
    HDS: hydrodesulphurisation — on/off per day (integer decision)

  Variables:
    x_it  ∈ [0, cap_i]      : throughput of continuous unit i on day t
    y_it  ∈ {0, 1}          : on/off status of integer units i on day t

  Objective:
    Maximise 5-day net margin = Σ_t Σ_i margin_it * x_it
                               - Σ_t Σ_i startup_cost_it * y_it

  Constraints:
    1. Capacity: x_it ≤ cap_i * y_it (if binary) or ≤ cap_i (if continuous)
    2. Feed balance: VDU feed ≤ CDU throughput
    3. FCC feed ≤ VDU throughput + FCC standalone capacity
    4. HDS must process all sulphur-rich streams from FCC
    5. Minimum run length: if FCC starts (y_t=1, y_{t-1}=0), must run ≥ 2 days
    6. Shutdown cost: HDS startup costs Rs 5,00,000 per start

This is an MILP.  We solve it with our branch-and-bound engine.

Results printed and saved to benchmarks/results/refinery_schedule_result.csv.
"""

from __future__ import annotations

import csv
import os
import numpy as np
import scipy.sparse as sp

from solver.milp.branch_and_bound import solve_milp
from solver.problem import Problem


# ── Problem data ──────────────────────────────────────────────────────────────

DAYS  = 5
UNITS = ["CDU", "VDU", "FCC", "HDS"]
N_UNITS = len(UNITS)

# Capacities (kL/day) — units 2 (FCC) and 3 (HDS) are on/off
CAPACITY = np.array([8000.0, 5000.0, 3000.0, 4000.0])

# Net margin (Rs/kL throughput, per day per unit)
MARGIN = np.array([
    [12.0, 11.5, 13.0, 12.5, 11.0],   # CDU
    [ 8.0,  8.5,  8.0,  9.0,  8.0],   # VDU
    [20.0, 21.0, 19.5, 22.0, 20.5],   # FCC (high value-add)
    [ 5.0,  5.0,  5.5,  5.0,  5.0],   # HDS
])

# Startup costs (Rs) for binary units (FCC, HDS) — only when y_{t}=1, y_{t-1}=0
STARTUP_COST = {"FCC": 200_000.0, "HDS": 500_000.0}

# Feed ratios: VDU feed = 0.6 * CDU throughput (at most)
VDU_FEED_FRACTION = 0.6
# FCC feed = VDU throughput + 500 kL from elsewhere (but ≤ 3000 cap)
FCC_STANDALONE = 500.0


def build_schedule_problem() -> Problem:
    """
    Variables (flat vector, row-major over units × days):
      x[i, t]  for i=0..3, t=0..4  →  index i*DAYS + t   (20 variables)
      y_fcc[t] for t=0..4           →  index 20 + t       (5 binary variables)
      y_hds[t] for t=0..4           →  index 25 + t       (5 binary variables)

    Total: n = 30 variables (20 continuous + 10 binary)
    """
    n_x = N_UNITS * DAYS       # 20 continuous throughput variables
    n_y = 2 * DAYS             # 10 binary on/off variables (FCC + HDS)
    n   = n_x + n_y

    int_mask = np.zeros(n, dtype=bool)
    int_mask[n_x:] = True      # last 10 variables are binary

    # ── Objective: maximise total net margin - startup costs ──────────────────
    c_obj = np.zeros(n)
    for i in range(N_UNITS):
        for t in range(DAYS):
            c_obj[i * DAYS + t] = -MARGIN[i, t]  # minimise negative margin

    # Startup cost penalisation: approximate as c_y for binary variables
    # (Full startup cost needs binary product constraints; here we use
    #  a simplified linear approximation: cost / max_capacity)
    for t in range(DAYS):
        c_obj[n_x + t]      = STARTUP_COST["FCC"] / CAPACITY[2]  # FCC on/off
        c_obj[n_x + DAYS + t] = STARTUP_COST["HDS"] / CAPACITY[3]  # HDS on/off

    # ── Constraints ────────────────────────────────────────────────────────────
    rows_A, rows_b = [], []

    for t in range(DAYS):
        idx_cdu = 0 * DAYS + t
        idx_vdu = 1 * DAYS + t
        idx_fcc = 2 * DAYS + t
        idx_hds = 3 * DAYS + t
        idx_yfcc = n_x + t
        idx_yhds = n_x + DAYS + t

        # 1a. Capacity CDU (continuous): x_cdu ≤ cap_cdu
        r = np.zeros(n); r[idx_cdu] = 1.0
        rows_A.append(r); rows_b.append(CAPACITY[0])

        # 1b. Capacity VDU (continuous): x_vdu ≤ cap_vdu
        r = np.zeros(n); r[idx_vdu] = 1.0
        rows_A.append(r); rows_b.append(CAPACITY[1])

        # 1c. FCC throughput ≤ cap_fcc * y_fcc: x_fcc - cap_fcc * y_fcc ≤ 0
        r = np.zeros(n); r[idx_fcc] = 1.0; r[idx_yfcc] = -CAPACITY[2]
        rows_A.append(r); rows_b.append(0.0)

        # 1d. HDS throughput ≤ cap_hds * y_hds
        r = np.zeros(n); r[idx_hds] = 1.0; r[idx_yhds] = -CAPACITY[3]
        rows_A.append(r); rows_b.append(0.0)

        # 2. VDU feed ≤ VDU_FEED_FRACTION * CDU throughput:
        #    x_vdu - VDU_FEED_FRACTION * x_cdu ≤ 0
        r = np.zeros(n); r[idx_vdu] = 1.0; r[idx_cdu] = -VDU_FEED_FRACTION
        rows_A.append(r); rows_b.append(0.0)

        # 3. FCC feed ≤ VDU throughput + FCC_STANDALONE:
        #    x_fcc ≤ x_vdu + FCC_STANDALONE
        #    x_fcc - x_vdu ≤ FCC_STANDALONE
        r = np.zeros(n); r[idx_fcc] = 1.0; r[idx_vdu] = -1.0
        rows_A.append(r); rows_b.append(FCC_STANDALONE)

        # 4. HDS must process at least FCC output (sulphur streams):
        #    x_hds ≥ 0.4 * x_fcc  →  -x_hds + 0.4 * x_fcc ≤ 0
        r = np.zeros(n); r[idx_hds] = -1.0; r[idx_fcc] = 0.4
        rows_A.append(r); rows_b.append(0.0)

    # 5. Min run length for FCC: if y_fcc[t] - y_fcc[t-1] = 1 (startup),
    #    then y_fcc[t+1] ≥ 1.  Approximated as:
    #    y_fcc[t-1] - y_fcc[t] - y_fcc[t+1] ≤ 0  (if y_t=0 after y_{t-1}=1)
    #    → linearised min-run: y_fcc[t] ≥ y_fcc[t-1] - y_fcc[t+1]
    #    → y_fcc[t-1] - y_fcc[t] - y_fcc[t+1] ≤ 0
    for t in range(1, DAYS - 1):
        r = np.zeros(n)
        r[n_x + (t-1)] =  1.0   # y_fcc[t-1]
        r[n_x + t]     = -1.0   # y_fcc[t]
        r[n_x + (t+1)] = -1.0   # y_fcc[t+1]
        rows_A.append(r); rows_b.append(0.0)

    A_ub = sp.csr_matrix(np.array(rows_A))
    b_ub = np.array(rows_b)

    lb = np.zeros(n)
    ub = np.concatenate([
        np.tile(CAPACITY, DAYS).reshape(N_UNITS, DAYS).T.ravel(),  # throughput bounds
        np.ones(n_y),  # binary y ∈ {0,1}
    ])

    return Problem(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=lb, ub=ub,
        integer_mask=int_mask,
        name="refinery_schedule_milp",
    )


def print_results(result):
    n_x = N_UNITS * DAYS
    if result.status not in ("optimal", "node_limit"):
        print(f"Solver status: {result.status}")
        return

    x = result.x
    x_th = x[:n_x].reshape(N_UNITS, DAYS)
    y_fcc = x[n_x:n_x + DAYS]
    y_hds = x[n_x + DAYS:]

    print("\n" + "=" * 70)
    print("   REFINERY UNIT SCHEDULING MILP — OPTIMAL SCHEDULE")
    print("=" * 70)
    print(f"Status: {result.status}  |  Nodes: {result.nodes}  |  LPR: {result.lp_relaxation:.0f}")
    print()
    header = f"{'Unit':<8}" + "".join(f"  Day{t+1:2d}" for t in range(DAYS))
    print(header)
    print("-" * 50)
    for i, unit in enumerate(UNITS):
        row = f"{unit:<8}" + "".join(f"  {x_th[i,t]:6.0f}" for t in range(DAYS))
        print(row)
    print()
    print("FCC on/off:  " + "  ".join(f"{int(round(y_fcc[t]))}" for t in range(DAYS)))
    print("HDS on/off:  " + "  ".join(f"{int(round(y_hds[t]))}" for t in range(DAYS)))
    print()

    # Net margin calculation (approximate)
    margin_total = -result.objective  # We minimised -margin
    print(f"5-day estimated net margin: Rs {margin_total * 1e-3:>10.1f}k")
    print("=" * 70)


def save_csv(result, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_x = N_UNITS * DAYS
    x = result.x if result.x is not None else np.zeros(30)
    x_th = x[:n_x].reshape(N_UNITS, DAYS)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "nodes", "objective"])
        writer.writerow([result.status, result.nodes, f"{-result.objective:.0f}"])
        writer.writerow([])
        writer.writerow(["unit"] + [f"day{t+1}" for t in range(DAYS)])
        for i, unit in enumerate(UNITS):
            writer.writerow([unit] + [f"{x_th[i, t]:.1f}" for t in range(DAYS)])


if __name__ == "__main__":
    print("Building refinery scheduling MILP...")
    problem = build_schedule_problem()
    print(f"  n_vars={problem.n_vars}, n_ineq={problem.n_ineq}")
    print(f"  n_integer={problem.integer_mask.sum()}")
    print("Solving...")
    result = solve_milp(problem, time_limit=60.0, verbose=False)
    print_results(result)
    out = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results",
                       "refinery_schedule_result.csv")
    save_csv(result, os.path.abspath(out))
    print(f"\nResult saved to: benchmarks/results/refinery_schedule_result.csv")
