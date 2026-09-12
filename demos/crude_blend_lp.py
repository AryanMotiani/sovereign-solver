"""
demos/crude_blend_lp.py
------------------------
Demo 1: Crude Oil Blending LP (Refinery Planning)

Problem:
  An Indian refinery can blend 3 crude grades to produce 2 products:
    - Diesel (D) and Naphtha (N)

  Crude grades (volumes in kL/day):
    C1: light sweet  — high N yield, low sulphur
    C2: medium sour  — balanced yields, medium sulphur
    C3: heavy sour   — high D yield, high sulphur

  Blending yields:
              C1    C2    C3
    Diesel    0.30  0.45  0.60   (fraction going to diesel)
    Naphtha   0.50  0.35  0.25   (fraction going to naphtha)
    Residue   0.20  0.20  0.15   (remainder; not modelled as product)

  Revenue:
    Diesel:  Rs 55/kL
    Naphtha: Rs 40/kL

  Costs:
    C1: Rs 70/kL,   C2: Rs 60/kL,   C3: Rs 55/kL

  Constraints:
    Sulphur limit: 0.5%C1 + 1.0%C2 + 2.5%C3 ≤ 1.5% (volume-weighted)
    Capacity: total crude throughput ≤ 10,000 kL/day
    Min diesel: ≥ 3,000 kL/day
    All volumes ≥ 0

Objective: Maximise net profit = Revenue(Diesel + Naphtha) - Cost(Crudes)

This is modelled as an LP in standard form and solved with solve_lp_revised.
Results are printed and saved to benchmarks/results/crude_blend_result.csv.
"""

from __future__ import annotations

import os
import csv
import numpy as np
import scipy.sparse as sp

from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem

# ── Problem data ──────────────────────────────────────────────────────────────

CRUDE_COST  = np.array([70.0, 60.0, 55.0])   # Rs/kL
DIESEL_REV  = 55.0   # Rs/kL
NAPHTHA_REV = 40.0   # Rs/kL

DIESEL_YIELD  = np.array([0.30, 0.45, 0.60])
NAPHTHA_YIELD = np.array([0.50, 0.35, 0.25])
SULPHUR_PCT   = np.array([0.5,  1.0,  2.5])   # %

MAX_THROUGHPUT = 10_000.0   # kL/day
MIN_DIESEL     = 3_000.0    # kL/day
MAX_SULPHUR    = 1.5        # % (volume-weighted average)


def build_blend_problem() -> Problem:
    """
    Variables: x = [C1, C2, C3]  (kL/day of each crude)
    """
    n = 3

    # Objective: maximise profit = Σ (diesel_rev * dy_i + naphtha_rev * ny_i - cost_i) * x_i
    profit_per_kl = (
        DIESEL_REV  * DIESEL_YIELD
      + NAPHTHA_REV * NAPHTHA_YIELD
      - CRUDE_COST
    )
    c = -profit_per_kl  # minimise negative profit

    # Sulphur constraint: Σ sulphur_i * x_i ≤ MAX_SULPHUR * Σ x_i
    # → Σ (sulphur_i - MAX_SULPHUR) * x_i ≤ 0
    a_sulphur = SULPHUR_PCT - MAX_SULPHUR

    # Min diesel: Σ diesel_yield_i * x_i ≥ MIN_DIESEL
    # → -Σ diesel_yield_i * x_i ≤ -MIN_DIESEL
    a_diesel = -DIESEL_YIELD

    # Capacity: Σ x_i ≤ MAX_THROUGHPUT
    a_cap = np.ones(n)

    A_ub = np.vstack([a_sulphur, a_diesel, a_cap])
    b_ub = np.array([0.0, -MIN_DIESEL, MAX_THROUGHPUT])

    return Problem(
        c=c,
        A_ub=sp.csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=sp.csr_matrix((0, n)),
        b_eq=np.zeros(0),
        lb=np.zeros(n),
        ub=np.full(n, np.inf),
        integer_mask=np.zeros(n, dtype=bool),
        name="crude_blend_lp",
    )


def print_results(result, problem: Problem):
    if result.status != "optimal":
        print(f"Solver status: {result.status}")
        return

    x = result.x
    diesel_vol  = DIESEL_YIELD  @ x
    naphtha_vol = NAPHTHA_YIELD @ x
    throughput  = x.sum()
    sulphur_avg = (SULPHUR_PCT @ x) / max(throughput, 1e-9)

    revenue = DIESEL_REV * diesel_vol + NAPHTHA_REV * naphtha_vol
    cost    = CRUDE_COST @ x
    profit  = revenue - cost

    print("\n" + "=" * 56)
    print("   CRUDE OIL BLENDING LP — OPTIMAL SOLUTION")
    print("=" * 56)
    print(f"{'Crude C1 throughput:':<35} {x[0]:>10.1f} kL/day")
    print(f"{'Crude C2 throughput:':<35} {x[1]:>10.1f} kL/day")
    print(f"{'Crude C3 throughput:':<35} {x[2]:>10.1f} kL/day")
    print(f"{'Total throughput:':<35} {throughput:>10.1f} kL/day")
    print(f"{'Diesel production:':<35} {diesel_vol:>10.1f} kL/day")
    print(f"{'Naphtha production:':<35} {naphtha_vol:>10.1f} kL/day")
    print(f"{'Avg sulphur (vol-wt):':<35} {sulphur_avg:>10.3f} %")
    print("-" * 56)
    print(f"{'Revenue:':<35} Rs {revenue:>10,.0f}/day")
    print(f"{'Crude cost:':<35} Rs {cost:>10,.0f}/day")
    print(f"{'NET PROFIT:':<35} Rs {profit:>10,.0f}/day")
    print("=" * 56)
    print(f"Solver iterations: {result.iterations}")


def save_csv(result, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = result.x if result.x is not None else [None, None, None]
    diesel_vol  = float(DIESEL_YIELD  @ x) if result.x is not None else None
    naphtha_vol = float(NAPHTHA_YIELD @ x) if result.x is not None else None
    profit      = -result.objective if result.status == "optimal" else None

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "value", "unit"])
        writer.writerow(["status",   result.status, ""])
        writer.writerow(["C1_kL",    f"{x[0]:.2f}", "kL/day"])
        writer.writerow(["C2_kL",    f"{x[1]:.2f}", "kL/day"])
        writer.writerow(["C3_kL",    f"{x[2]:.2f}", "kL/day"])
        writer.writerow(["diesel_kL",  f"{diesel_vol:.2f}", "kL/day"])
        writer.writerow(["naphtha_kL", f"{naphtha_vol:.2f}", "kL/day"])
        writer.writerow(["profit_Rs",  f"{profit:.0f}", "Rs/day"])
        writer.writerow(["iters",    result.iterations, ""])


if __name__ == "__main__":
    problem = build_blend_problem()
    result  = solve_lp_revised(problem)
    print_results(result, problem)
    out = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results", "crude_blend_result.csv")
    save_csv(result, os.path.abspath(out))
    print(f"\nResult saved to: benchmarks/results/crude_blend_result.csv")
