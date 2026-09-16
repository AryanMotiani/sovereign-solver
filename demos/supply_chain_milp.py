"""
demos/supply_chain_milp.py
---------------------------
Supply Chain Network Optimization — Sovereign Solver Demo

Models a multi-echelon supply chain network: factories -> distribution
centers (DCs) -> customers, with binary facility opening decisions and
continuous flow variables.

Problem type: MILP

Background:
  This is a capacitated facility location + transportation problem, a
  fundamental model in Indian logistics and supply chain management used
  for distribution network design, warehouse location, and inventory routing.

Model:
  Given:
    F = factories (supply nodes)
    D = distribution centers (DCs, potential locations)
    C = customers (demand nodes)
    cap_f  = factory production capacity (units/period)
    cap_d  = DC throughput capacity (units/period)
    f_cost = DC fixed opening cost (Rs.)
    t_fd   = transport cost factory -> DC (Rs./unit)
    t_dc   = transport cost DC -> customer (Rs./unit)
    demand = customer demand (units/period)

  Variables:
    y_d       = 1 if DC d is opened (binary)
    x_fd      = units shipped from factory f to DC d (continuous)
    z_dc      = units shipped from DC d to customer c (continuous)

  Objective:
    min  Σ_d f_cost_d * y_d              (fixed opening cost)
       + Σ_f Σ_d t_fd[f,d] * x_fd       (factory -> DC transport)
       + Σ_d Σ_c t_dc[d,c] * z_dc       (DC -> customer transport)

  Constraints:
    1. Customer demand:     Σ_d z_dc >= demand_c   ∀c
    2. DC flow balance:    Σ_f x_fd = Σ_c z_dc     ∀d  (in = out)
    3. DC capacity:        Σ_c z_dc <= cap_d * y_d  ∀d  (only if open)
    4. Factory capacity:   Σ_d x_fd <= cap_f        ∀f
    5. Non-negativity:     x_fd, z_dc >= 0

Scenario:
  3 factories (Mumbai, Chennai, Delhi)
  5 potential DCs (Pune, Hyderabad, Nagpur, Jaipur, Kolkata)
  8 customers (major Indian cities)

Usage:
    python -m demos.supply_chain_milp
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp

from solver.problem import Problem
from solver.milp.branch_and_bound import solve_milp


# ── Network data ──────────────────────────────────────────────────────────────

FACTORIES = ["Mumbai", "Chennai", "Delhi"]
DCS = ["Pune", "Hyderabad", "Nagpur", "Jaipur", "Kolkata"]
CUSTOMERS = ["Bengaluru", "Ahmedabad", "Surat", "Lucknow",
             "Patna", "Bhopal", "Indore", "Visakhapatnam"]

# Factory production capacities (units/period)
FACTORY_CAP = {"Mumbai": 500, "Chennai": 400, "Delhi": 600}

# DC throughput capacities & fixed opening costs (Rs. thousands)
DC_CAP   = {"Pune": 300, "Hyderabad": 350, "Nagpur": 250, "Jaipur": 280, "Kolkata": 320}
DC_FCOST = {"Pune": 120, "Hyderabad": 100, "Nagpur":  80, "Jaipur":  90, "Kolkata": 110}

# Customer demands (units/period)
DEMAND = {
    "Bengaluru": 90, "Ahmedabad": 75, "Surat": 60, "Lucknow": 80,
    "Patna": 55, "Bhopal": 65, "Indore": 70, "Visakhapatnam": 50,
}

# Transport costs: factory -> DC  (Rs./unit, distance-proportional estimate)
T_FD = {
    ("Mumbai",  "Pune"):       2, ("Mumbai",  "Hyderabad"): 8,
    ("Mumbai",  "Nagpur"):     6, ("Mumbai",  "Jaipur"):    7,
    ("Mumbai",  "Kolkata"):   12,
    ("Chennai", "Pune"):       9, ("Chennai", "Hyderabad"): 3,
    ("Chennai", "Nagpur"):     7, ("Chennai", "Jaipur"):   11,
    ("Chennai", "Kolkata"):    8,
    ("Delhi",   "Pune"):       8, ("Delhi",   "Hyderabad"):10,
    ("Delhi",   "Nagpur"):     6, ("Delhi",   "Jaipur"):    4,
    ("Delhi",   "Kolkata"):    9,
}

# Transport costs: DC -> customer  (Rs./unit)
T_DC = {
    ("Pune",      "Bengaluru"): 5, ("Pune",      "Ahmedabad"): 3,
    ("Pune",      "Surat"):     4, ("Pune",      "Lucknow"):   8,
    ("Pune",      "Patna"):    11, ("Pune",      "Bhopal"):    6,
    ("Pune",      "Indore"):    5, ("Pune",      "Visakhapatnam"): 9,
    ("Hyderabad", "Bengaluru"): 3, ("Hyderabad", "Ahmedabad"): 8,
    ("Hyderabad", "Surat"):     7, ("Hyderabad", "Lucknow"):  10,
    ("Hyderabad", "Patna"):    10, ("Hyderabad", "Bhopal"):    6,
    ("Hyderabad", "Indore"):    7, ("Hyderabad", "Visakhapatnam"): 4,
    ("Nagpur",    "Bengaluru"): 7, ("Nagpur",    "Ahmedabad"): 6,
    ("Nagpur",    "Surat"):     5, ("Nagpur",    "Lucknow"):   7,
    ("Nagpur",    "Patna"):     8, ("Nagpur",    "Bhopal"):    3,
    ("Nagpur",    "Indore"):    4, ("Nagpur",    "Visakhapatnam"): 7,
    ("Jaipur",    "Bengaluru"):11, ("Jaipur",    "Ahmedabad"): 4,
    ("Jaipur",    "Surat"):     5, ("Jaipur",    "Lucknow"):   4,
    ("Jaipur",    "Patna"):     7, ("Jaipur",    "Bhopal"):    5,
    ("Jaipur",    "Indore"):    4, ("Jaipur",    "Visakhapatnam"):12,
    ("Kolkata",   "Bengaluru"): 9, ("Kolkata",   "Ahmedabad"):12,
    ("Kolkata",   "Surat"):    13, ("Kolkata",   "Lucknow"):   5,
    ("Kolkata",   "Patna"):     3, ("Kolkata",   "Bhopal"):    8,
    ("Kolkata",   "Indore"):    9, ("Kolkata",   "Visakhapatnam"): 5,
}


def build_supply_chain_problem() -> Problem:
    """
    Build the MILP for the supply chain network design.

    Variable ordering (flat):
      y[d]        : d=0..D-1           (D binary DC open/close)
      x[f,d]      : f=0..F-1, d=0..D-1 (F*D continuous factory->DC flows)
      z[d,c]      : d=0..D-1, c=0..C-1 (D*C continuous DC->customer flows)
    """
    F = len(FACTORIES)
    D = len(DCS)
    C = len(CUSTOMERS)

    n_y = D
    n_x = F * D
    n_z = D * C
    n = n_y + n_x + n_z

    def y_idx(d):        return d
    def x_idx(f, d):     return n_y + f * D + d
    def z_idx(d, c):     return n_y + n_x + d * C + c

    # ── Objective ──────────────────────────────────────────────────────────────
    c_obj = np.zeros(n)
    for d, dc in enumerate(DCS):
        c_obj[y_idx(d)] = DC_FCOST[dc]
    for f, fac in enumerate(FACTORIES):
        for d, dc in enumerate(DCS):
            c_obj[x_idx(f, d)] = T_FD[(fac, dc)]
    for d, dc in enumerate(DCS):
        for c, cust in enumerate(CUSTOMERS):
            c_obj[z_idx(d, c)] = T_DC[(dc, cust)]

    # ── Constraints ────────────────────────────────────────────────────────────
    rows_ub, cols_ub, data_ub, rhs_ub = [], [], [], []
    rows_eq, cols_eq, data_eq, rhs_eq = [], [], [], []

    row_ub = 0
    row_eq = 0

    def add_ub(terms, b):
        nonlocal row_ub
        for coef, idx in terms:
            rows_ub.append(row_ub); cols_ub.append(idx); data_ub.append(coef)
        rhs_ub.append(b); row_ub += 1

    def add_eq(terms, b):
        nonlocal row_eq
        for coef, idx in terms:
            rows_eq.append(row_eq); cols_eq.append(idx); data_eq.append(coef)
        rhs_eq.append(b); row_eq += 1

    # 1. Customer demand: Σ_d z[d,c] >= demand_c
    #    -> -Σ_d z[d,c] <= -demand_c
    for c, cust in enumerate(CUSTOMERS):
        terms = [(-1.0, z_idx(d, c)) for d in range(D)]
        add_ub(terms, -DEMAND[cust])

    # 2. DC flow balance: Σ_f x[f,d] - Σ_c z[d,c] = 0  ∀d
    for d in range(D):
        terms = (
            [(1.0, x_idx(f, d)) for f in range(F)] +
            [(-1.0, z_idx(d, c)) for c in range(C)]
        )
        add_eq(terms, 0.0)

    # 3. DC capacity: Σ_c z[d,c] - cap_d * y[d] <= 0  ∀d
    for d, dc in enumerate(DCS):
        terms = [(1.0, z_idx(d, c)) for c in range(C)]
        terms += [(-DC_CAP[dc], y_idx(d))]
        add_ub(terms, 0.0)

    # 4. Factory capacity: Σ_d x[f,d] <= cap_f  ∀f
    for f, fac in enumerate(FACTORIES):
        terms = [(1.0, x_idx(f, d)) for d in range(D)]
        add_ub(terms, FACTORY_CAP[fac])

    m_ub = row_ub
    m_eq = row_eq

    A_ub = sp.csr_matrix((data_ub, (rows_ub, cols_ub)), shape=(m_ub, n), dtype=float)
    A_eq = sp.csr_matrix((data_eq, (rows_eq, cols_eq)), shape=(m_eq, n), dtype=float)

    # ── Bounds ──────────────────────────────────────────────────────────────────
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    for d in range(D):
        ub[y_idx(d)] = 1.0  # binary

    # ── Integer mask ──────────────────────────────────────────────────────────
    int_mask = np.zeros(n, dtype=bool)
    for d in range(D):
        int_mask[y_idx(d)] = True

    return Problem(
        c=c_obj, A_ub=A_ub, b_ub=np.array(rhs_ub),
        A_eq=A_eq, b_eq=np.array(rhs_eq),
        lb=lb, ub=ub, integer_mask=int_mask,
        sense="min", name="supply_chain_india",
    )


def print_solution(result, prob: Problem) -> None:
    F, D, C = len(FACTORIES), len(DCS), len(CUSTOMERS)
    n_y = D
    n_x = F * D
    x = result.x
    if x is None:
        return

    print("\n  Distribution Centers Opened:")
    for d, dc in enumerate(DCS):
        opened = round(x[d])
        if opened:
            print(f"    {dc:12s} (cap={DC_CAP[dc]:3d}u, cost=Rs.{DC_FCOST[dc]}k)")

    print("\n  Factory -> DC Flows (units > 0):")
    for f, fac in enumerate(FACTORIES):
        for d, dc in enumerate(DCS):
            flow = x[n_y + f * D + d]
            if flow > 0.5:
                print(f"    {fac:8s} -> {dc:12s}: {flow:.1f} units  "
                      f"(cost Rs.{T_FD[(fac,dc)]}/u)")

    print("\n  DC -> Customer Flows (units > 0):")
    for d, dc in enumerate(DCS):
        for c, cust in enumerate(CUSTOMERS):
            flow = x[n_y + n_x + d * C + c]
            if flow > 0.5:
                print(f"    {dc:12s} -> {cust:14s}: {flow:.1f} units")


def run_demo() -> None:
    print("=" * 70)
    print("  SUPPLY CHAIN NETWORK OPTIMIZATION — SOVEREIGN SOLVER DEMO")
    print(f"  {len(FACTORIES)} Factories, {len(DCS)} potential DCs, {len(CUSTOMERS)} Customers")
    print("=" * 70)

    total_demand = sum(DEMAND.values())
    print(f"\n  Total demand: {total_demand} units/period")
    print(f"  Total factory capacity: {sum(FACTORY_CAP.values())} units/period")

    prob = build_supply_chain_problem()
    print(f"\n  Problem: {prob.n_vars} variables "
          f"({prob.integer_mask.sum()} binary DC decisions + "
          f"{prob.n_vars - prob.integer_mask.sum()} continuous flows)")
    print(f"           {prob.n_ineq} inequality + {prob.n_eq} equality constraints")

    print("\n  Solving with Branch-and-Cut...")
    t0 = time.perf_counter()
    result = solve_milp(prob, time_limit=120.0, verbose=False)
    elapsed = time.perf_counter() - t0

    print(f"\n  Status:    {result.status}")
    print(f"  Total Cost: Rs.{result.objective:.1f}k/period")
    print(f"  LP Bound:   Rs.{result.lp_relaxation:.1f}k/period")
    print(f"  Gap:        {result.gap*100:.2f}%")
    print(f"  B&B Nodes:  {result.nodes}")
    print(f"  Wall Time:  {elapsed:.2f}s")

    if result.x is not None:
        print_solution(result, prob)

        # Verify all demands met
        F, D, C = len(FACTORIES), len(DCS), len(CUSTOMERS)
        n_y, n_x = D, F * D
        x = result.x
        print("\n  Demand verification:")
        all_ok = True
        for c, cust in enumerate(CUSTOMERS):
            delivered = sum(x[n_y + n_x + d * C + c] for d in range(D))
            req = DEMAND[cust]
            ok = delivered >= req - 1e-3
            if not ok:
                all_ok = False
            print(f"    {cust:14s}: {delivered:.1f}/{req} units  {'OK' if ok else 'SHORTFALL'}")
        if all_ok:
            print("\n  All customer demands satisfied.")


if __name__ == "__main__":
    run_demo()
