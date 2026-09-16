"""
demos/transportation_lp.py
---------------------------
Transportation Problem — Sovereign Solver Demo

Models the classical Hitchcock-Koopmans transportation problem:
Find the minimum-cost shipping plan from multiple supply nodes to multiple
demand nodes.

Problem type: LP (network flow — all-integer optimal at LP vertex)

Background:
  Transportation problems arise in road freight planning, railway goods
  movement, port logistics, and inter-state commodity distribution.
  They have a totally unimodular constraint matrix, so LP relaxation
  always yields integer solutions.

Models included:
  1. Basic transportation LP (Hitchcock-Koopmans)
  2. Minimum-cost flow on a road network (extended)
  3. Multi-commodity transportation (two commodities)

Scenario (India Rail Freight):
  Supply nodes: 5 major freight loading terminals
    (Mundra Port, JNPT Mumbai, Haldia, Vizag Port, Kochi)
  Demand nodes: 6 inland container depots
    (Ludhiana, Moradabad, Nagpur, Agra, Ahmedabad, Coimbatore)

Usage:
    python -m demos.transportation_lp
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp

from solver.problem import Problem
from solver.lp.simplex_revised import solve_lp_revised


# ── Scenario Data ─────────────────────────────────────────────────────────────

PORTS = ["Mundra", "JNPT_Mumbai", "Haldia", "Vizag", "Kochi"]
DEPOTS = ["Ludhiana", "Moradabad", "Nagpur", "Agra", "Ahmedabad", "Coimbatore"]

# Supply (TEUs per week)
SUPPLY = {
    "Mundra":      800,
    "JNPT_Mumbai": 1200,
    "Haldia":       600,
    "Vizag":        500,
    "Kochi":        400,
}

# Demand (TEUs per week)
DEMAND = {
    "Ludhiana":    400,
    "Moradabad":   350,
    "Nagpur":      450,
    "Agra":        300,
    "Ahmedabad":   600,
    "Coimbatore":  400,
}

# Transport cost (Rs. thousands per TEU) — symmetric rail/road blended rate
COST = {
    ("Mundra",      "Ludhiana"):    12, ("Mundra",      "Moradabad"):   14,
    ("Mundra",      "Nagpur"):       9, ("Mundra",      "Agra"):        13,
    ("Mundra",      "Ahmedabad"):    4, ("Mundra",      "Coimbatore"):  18,
    ("JNPT_Mumbai", "Ludhiana"):    15, ("JNPT_Mumbai", "Moradabad"):   16,
    ("JNPT_Mumbai", "Nagpur"):       7, ("JNPT_Mumbai", "Agra"):        14,
    ("JNPT_Mumbai", "Ahmedabad"):    5, ("JNPT_Mumbai", "Coimbatore"):  14,
    ("Haldia",      "Ludhiana"):    16, ("Haldia",      "Moradabad"):   12,
    ("Haldia",      "Nagpur"):      11, ("Haldia",      "Agra"):        13,
    ("Haldia",      "Ahmedabad"):   18, ("Haldia",      "Coimbatore"):  20,
    ("Vizag",       "Ludhiana"):    19, ("Vizag",       "Moradabad"):   17,
    ("Vizag",       "Nagpur"):       8, ("Vizag",       "Agra"):        16,
    ("Vizag",       "Ahmedabad"):   15, ("Vizag",       "Coimbatore"):  10,
    ("Kochi",       "Ludhiana"):    22, ("Kochi",       "Moradabad"):   21,
    ("Kochi",       "Nagpur"):      12, ("Kochi",       "Agra"):        20,
    ("Kochi",       "Ahmedabad"):   16, ("Kochi",       "Coimbatore"):   5,
}


def build_transportation_problem(
    supply: dict, demand: dict, cost: dict,
    ports: list, depots: list,
    name: str = "transport",
) -> Problem:
    """
    Build the transportation LP.

    min   Σ_ij c_ij * x_ij
    s.t.  Σ_j x_ij  = supply_i   ∀i    (supply constraints — equality)
          Σ_i x_ij  = demand_j   ∀j    (demand constraints — equality)
          x_ij >= 0

    For a balanced problem (Σ supply = Σ demand), all constraints are equalities.
    For an unbalanced problem, supply constraints become <=.
    """
    S = len(ports)
    D_n = len(depots)
    n = S * D_n  # x[i,j] -> flat index i*D_n + j

    # Check balance
    total_supply = sum(supply.values())
    total_demand = sum(demand.values())
    balanced = abs(total_supply - total_demand) < 1e-6

    c_obj = np.array([
        cost[(ports[i], depots[j])]
        for i in range(S) for j in range(D_n)
    ], dtype=float)

    # Build constraint matrix
    # Supply rows: Σ_j x[i,j] = supply_i  or <= supply_i
    # Demand rows: Σ_i x[i,j] = demand_j
    eq_rows, eq_cols, eq_data, eq_rhs = [], [], [], []
    ub_rows, ub_cols, ub_data, ub_rhs = [], [], [], []

    eq_r = 0; ub_r = 0

    if balanced:
        # Supply as equalities
        for i, port in enumerate(ports):
            for j in range(D_n):
                eq_rows.append(eq_r); eq_cols.append(i * D_n + j); eq_data.append(1.0)
            eq_rhs.append(supply[port]); eq_r += 1
        # Demand as equalities
        for j, depot in enumerate(depots):
            for i in range(S):
                eq_rows.append(eq_r); eq_cols.append(i * D_n + j); eq_data.append(1.0)
            eq_rhs.append(demand[depot]); eq_r += 1
    else:
        # Supply as inequalities (≤)
        for i, port in enumerate(ports):
            for j in range(D_n):
                ub_rows.append(ub_r); ub_cols.append(i * D_n + j); ub_data.append(1.0)
            ub_rhs.append(supply[port]); ub_r += 1
        # Demand as equalities
        for j, depot in enumerate(depots):
            for i in range(S):
                eq_rows.append(eq_r); eq_cols.append(i * D_n + j); eq_data.append(1.0)
            eq_rhs.append(demand[depot]); eq_r += 1

    A_eq = sp.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(eq_r, n), dtype=float)
    A_ub = sp.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(ub_r, n), dtype=float)

    return Problem(
        c=c_obj, A_ub=A_ub, b_ub=np.array(ub_rhs, float),
        A_eq=A_eq, b_eq=np.array(eq_rhs, float),
        lb=np.zeros(n), ub=np.full(n, np.inf),
        integer_mask=np.zeros(n, dtype=bool),
        sense="min", name=name,
    )


def print_transport_solution(result, ports: list, depots: list) -> None:
    S, D_n = len(ports), len(depots)
    x = result.x
    if x is None:
        return
    print("\n  Optimal shipping plan (TEUs/week):")
    print("  " + " " * 14 + "".join(f"{d[:8]:>10}" for d in depots))
    print("  " + "-" * (14 + 10 * D_n))
    for i, port in enumerate(ports):
        row = "".join(
            f"{x[i*D_n+j]:>10.0f}" for j in range(D_n)
        )
        total = sum(x[i*D_n+j] for j in range(D_n))
        print(f"  {port:14s}{row}  | {total:.0f}")
    print("  " + "-" * (14 + 10 * D_n))
    totals = "".join(f"{sum(x[i*D_n+j] for i in range(S)):>10.0f}" for j in range(D_n))
    print(f"  {'TOTAL':14s}{totals}")


def run_basic_demo() -> None:
    """Classical Hitchcock-Koopmans transportation LP."""
    print("=" * 70)
    print("  TRANSPORTATION PROBLEM — SOVEREIGN SOLVER DEMO")
    print("  India Rail Freight: 5 Ports -> 6 Inland Depots")
    print("=" * 70)

    ts = sum(SUPPLY.values())
    td = sum(DEMAND.values())
    print(f"\n  Total supply: {ts} TEUs/week,  Total demand: {td} TEUs/week")
    balanced = abs(ts - td) < 1
    print(f"  Balanced: {'Yes' if balanced else 'No (slack/surplus)'}")

    prob = build_transportation_problem(SUPPLY, DEMAND, COST, PORTS, DEPOTS)
    print(f"\n  Problem: {prob.n_vars} variables, "
          f"{prob.n_eq} equality + {prob.n_ineq} inequality constraints")

    t0 = time.perf_counter()
    result = solve_lp_revised(prob)
    elapsed = time.perf_counter() - t0

    print(f"\n  Status:     {result.status}")
    print(f"  Total Cost: Rs.{result.objective:.1f}k/week")
    print(f"  Iterations: {result.iterations}")
    print(f"  Wall Time:  {elapsed:.4f}s")

    if result.x is not None:
        print_transport_solution(result, PORTS, DEPOTS)

        # Integrality check (TU matrix -> always integer at LP optimum)
        x = result.x
        frac_count = np.sum(np.abs(x - np.round(x)) > 1e-4)
        print(f"\n  Integer check: {frac_count} fractional values "
              f"(expect 0 — TU matrix)")


def run_multi_commodity_demo() -> None:
    """Two-commodity transportation on the same network."""
    print("\n" + "=" * 70)
    print("  MULTI-COMMODITY TRANSPORTATION DEMO")
    print("  Commodity A (steel), Commodity B (fertilizer)")
    print("=" * 70)

    # Commodity A: steel
    supply_a = {"Mundra": 300, "JNPT_Mumbai": 400, "Haldia": 200,
                 "Vizag": 150, "Kochi": 100}
    demand_a = {"Ludhiana": 150, "Moradabad": 120, "Nagpur": 180,
                 "Agra": 130, "Ahmedabad": 270, "Coimbatore": 100}
    cost_a = {k: v + 2 for k, v in COST.items()}  # slightly higher for steel

    # Commodity B: fertilizer
    supply_b = {"Mundra": 500, "JNPT_Mumbai": 800, "Haldia": 400,
                 "Vizag": 350, "Kochi": 300}
    demand_b = {"Ludhiana": 250, "Moradabad": 230, "Nagpur": 270,
                 "Agra": 170, "Ahmedabad": 330, "Coimbatore": 300}
    cost_b = COST.copy()

    for name, supply, demand, cost in [
        ("Steel", supply_a, demand_a, cost_a),
        ("Fertilizer", supply_b, demand_b, cost_b),
    ]:
        prob = build_transportation_problem(supply, demand, cost, PORTS, DEPOTS,
                                           name=f"transport_{name.lower()}")
        result = solve_lp_revised(prob)
        ts = sum(supply.values()); td = sum(demand.values())
        print(f"\n  {name}: supply={ts}, demand={td}")
        if abs(ts - td) > 1:
            print(f"  (Unbalanced by {abs(ts-td)} — excess supply)")
        print(f"  Status: {result.status}, Cost: Rs.{result.objective:.1f}k/week, "
              f"Iters: {result.iterations}")


if __name__ == "__main__":
    run_basic_demo()
    run_multi_commodity_demo()
