"""
demos/power_economic_dispatch.py
---------------------------------
Power System Economic Dispatch — Sovereign Solver Demo

Models the Unit Commitment + Economic Dispatch problem for a
small power system with 6 thermal generation units.

Problem type: MILP (unit commitment binary variables + continuous dispatch)

Background:
  Economic dispatch decides the optimal power output of generators to
  meet demand at minimum cost. Unit commitment additionally decides
  which generators to switch on/off each hour.

Model:
  Given:
    T = 24 time periods (hours)
    N = 6 generating units
    D[t] = demand at time t  (MW)
    C_i = fuel cost ($/MWh)
    SC_i = startup cost ($)
    P_min_i, P_max_i = min/max output (MW) when on
    RU_i, RD_i = ramp-up/ramp-down limits (MW/hour)
    Reserve = spinning reserve requirement (10% of demand)

  Variables:
    p[i,t]  = power output of unit i at time t  (continuous, MW)
    u[i,t]  = on/off status of unit i at time t  (binary)
    v[i,t]  = startup indicator (binary: 1 if unit i starts up at t)

  Objective:
    min ΣΣ C_i * p[i,t] + ΣΣ SC_i * v[i,t]

  Constraints:
    1. Demand: Σ_i p[i,t] >= D[t]  ∀t
    2. Spinning reserve: Σ_i u[i,t]*P_max_i >= (1+0.1)*D[t]  ∀t
    3. Output bounds: P_min_i*u[i,t] <= p[i,t] <= P_max_i*u[i,t]  ∀i,t
    4. Ramp up: p[i,t] - p[i,t-1] <= RU_i  ∀i,t>1
    5. Ramp down: p[i,t-1] - p[i,t] <= RD_i  ∀i,t>1
    6. Startup: v[i,t] >= u[i,t] - u[i,t-1]  ∀i,t>1

We linearize bound constraints 3 (split into two inequalities):
    p[i,t] <= P_max_i * u[i,t]
   -p[i,t] <= -P_min_i * u[i,t]   (since u is binary, we substitute directly)

Due to MILP size (6*24*3 = 432 vars, ~500 constraints), we use a 2-hour
horizon for the demo to keep runtime manageable for demonstration purposes.

Usage:
    python demos/power_economic_dispatch.py
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp

from solver.problem import Problem
from solver.milp.branch_and_bound import solve_milp


# ── Unit data ─────────────────────────────────────────────────────────────────

# 6 thermal units: [coal1, coal2, gas1, gas2, oil1, oil2]
UNITS = {
    "name":     ["Coal-1",  "Coal-2",  "Gas-1",   "Gas-2",   "Oil-1",  "Oil-2"],
    "P_max":    [400.0,     350.0,     200.0,     150.0,     100.0,    80.0],   # MW
    "P_min":    [100.0,     80.0,      50.0,      40.0,      25.0,     20.0],   # MW
    "C_fuel":   [20.0,      22.0,      35.0,      38.0,      55.0,     60.0],   # $/MWh
    "C_start":  [1500.0,    1200.0,    600.0,     500.0,     300.0,    250.0],  # $/startup
    "RU":       [80.0,      70.0,      100.0,     80.0,      50.0,     40.0],   # MW/h
    "RD":       [80.0,      70.0,      100.0,     80.0,      50.0,     40.0],   # MW/h
}

# Demand profile for T hours (MW) — representative morning ramp-up
DEMAND_FULL = [
    300, 280, 270, 260, 280, 350,  # 0–5: overnight + early morning
    450, 600, 700, 750, 780, 800,  # 6–11: morning ramp
    790, 800, 780, 750, 720, 700,  # 12–17: afternoon
    680, 700, 720, 680, 600, 450,  # 18–23: evening
]


def build_dispatch_problem(T: int = 4) -> Problem:
    """
    Build the unit commitment + economic dispatch MILP for T time periods.

    Variable ordering (flat index):
      [p_{0,0}, p_{1,0}, ..., p_{N-1,0},  ← continuous outputs hour 0
       p_{0,1}, ...,                        ← hour 1
       ...,
       u_{0,0}, u_{1,0}, ..., u_{N-1,0},  ← binary on/off hour 0
       ...,
       v_{0,1}, ..., v_{N-1,1},            ← binary startup hour 1
       ...]

    Returns Problem with integer_mask set for all u and v variables.
    """
    N = len(UNITS["P_max"])
    P_max = np.array(UNITS["P_max"])
    P_min = np.array(UNITS["P_min"])
    C_fuel = np.array(UNITS["C_fuel"])
    C_start = np.array(UNITS["C_start"])
    RU = np.array(UNITS["RU"])
    RD = np.array(UNITS["RD"])
    demand = np.array(DEMAND_FULL[:T])

    # Variable blocks
    # p[i, t] : index = t*N + i                 (N*T continuous)
    # u[i, t] : index = N*T + t*N + i           (N*T binary)
    # v[i, t] : index = N*T + N*T + (t-1)*N+i  (N*(T-1) binary, t=1..T-1)

    n_p = N * T
    n_u = N * T
    n_v = N * (T - 1)
    n = n_p + n_u + n_v

    # ── Objective ──────────────────────────────────────────────────────────────
    c_obj = np.zeros(n)
    for t in range(T):
        for i in range(N):
            c_obj[t * N + i] = C_fuel[i]                    # fuel cost * output
    for t in range(1, T):
        for i in range(N):
            v_idx = n_p + n_u + (t - 1) * N + i
            c_obj[v_idx] = C_start[i]                        # startup cost

    # ── Constraints ────────────────────────────────────────────────────────────
    rows_ub, cols_ub, data_ub, rhs_ub = [], [], [], []
    rows_eq, cols_eq, data_eq, rhs_eq = [], [], [], []

    row_ub = 0

    def add_ub(coeffs_idx_val, b):
        nonlocal row_ub
        for coef, idx in coeffs_idx_val:
            rows_ub.append(row_ub)
            cols_ub.append(int(idx))
            data_ub.append(float(coef))
        rhs_ub.append(b)
        row_ub += 1

    for t in range(T):
        # 1. Demand: -Σ_i p[i,t] <= -D[t]
        add_ub([(-1.0, t * N + i) for i in range(N)], -demand[t])

        # 2. Reserve: -Σ_i P_max_i * u[i,t] <= -1.1 * D[t]
        add_ub(
            [(-P_max[i], n_p + t * N + i) for i in range(N)],
            -1.1 * demand[t]
        )

        for i in range(N):
            p_idx = t * N + i
            u_idx = n_p + t * N + i

            # 3a. Output upper: p[i,t] - P_max_i * u[i,t] <= 0
            add_ub([(1.0, p_idx), (-P_max[i], u_idx)], 0.0)

            # 3b. Output lower: -p[i,t] + P_min_i * u[i,t] <= 0
            add_ub([(-1.0, p_idx), (P_min[i], u_idx)], 0.0)

        if t > 0:
            for i in range(N):
                p_cur = t * N + i
                p_prv = (t - 1) * N + i
                v_idx = n_p + n_u + (t - 1) * N + i

                # 4. Ramp up: p[i,t] - p[i,t-1] <= RU_i
                add_ub([(1.0, p_cur), (-1.0, p_prv)], RU[i])

                # 5. Ramp down: p[i,t-1] - p[i,t] <= RD_i
                add_ub([(1.0, p_prv), (-1.0, p_cur)], RD[i])

                # 6. Startup: v[i,t] >= u[i,t] - u[i,t-1]
                #    → -v[i,t] + u[i,t] - u[i,t-1] <= 0
                u_cur = n_p + t * N + i
                u_prv = n_p + (t - 1) * N + i
                add_ub(
                    [(-1.0, v_idx), (1.0, u_cur), (-1.0, u_prv)],
                    0.0
                )

    m_ub = row_ub
    A_ub = sp.csr_matrix(
        (data_ub, (rows_ub, cols_ub)), shape=(m_ub, n), dtype=float
    )
    b_ub = np.array(rhs_ub, dtype=float)

    # ── Bounds ──────────────────────────────────────────────────────────────────
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    # Continuous outputs: lb=0, ub=P_max
    for t in range(T):
        for i in range(N):
            ub[t * N + i] = P_max[i]
    # Binary on/off
    for t in range(T):
        for i in range(N):
            ub[n_p + t * N + i] = 1.0
    # Binary startup
    for t in range(T - 1):
        for i in range(N):
            ub[n_p + n_u + t * N + i] = 1.0

    # ── Integer mask ──────────────────────────────────────────────────────────
    int_mask = np.zeros(n, dtype=bool)
    int_mask[n_p:] = True   # all u and v are binary

    return Problem(
        c=c_obj,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=sp.csr_matrix((0, n)), b_eq=np.zeros(0),
        lb=lb, ub=ub,
        integer_mask=int_mask,
        sense="min",
        name=f"economic_dispatch_T{T}",
    )


def print_schedule(result, T: int, N: int = 6) -> None:
    """Print a human-readable unit commitment schedule."""
    n_p = N * T
    n_u = N * T
    x = result.x
    names = UNITS["name"]

    print("\n  Hour  | " + " | ".join(f"{nm:>8}" for nm in names) + "  | Total MW")
    print("  " + "-" * (8 + 13 * N))
    for t in range(T):
        outs = [x[t * N + i] for i in range(N)]
        on   = [int(round(x[n_p + t * N + i])) for i in range(N)]
        total = sum(outs)
        cells = " | ".join(
            f"{outs[i]:>5.0f}{'*' if on[i] else ' ':>2s}"
            for i in range(N)
        )
        print(f"  {t:4d}  | {cells}  | {total:7.1f}")
    print("  (* = unit on)")


def run_demo(T: int = 4) -> None:
    print("=" * 70)
    print("  POWER SYSTEM ECONOMIC DISPATCH — SOVEREIGN SOLVER DEMO")
    print(f"  T={T} hours, N=6 thermal units")
    print("=" * 70)

    demand = DEMAND_FULL[:T]
    print(f"\n  Demand profile: {demand} MW")
    print(f"  Total demand: {sum(demand):.0f} MWh")

    prob = build_dispatch_problem(T)
    N = 6
    n_p = N * T
    n_u = N * T
    n_v = N * (T - 1)
    print(f"\n  Problem size: {prob.n_vars} variables ({n_p} continuous, {n_u+n_v} binary)")
    print(f"               {prob.n_ineq} inequality constraints")
    print(f"               {prob.integer_mask.sum()} integer variables")

    print("\n  Solving with Branch-and-Cut...")
    t0 = time.perf_counter()
    result = solve_milp(prob, time_limit=120.0, verbose=False)
    elapsed = time.perf_counter() - t0

    print(f"\n  Status:    {result.status}")
    print(f"  Objective: ${result.objective:,.2f}  (total fuel + startup cost)")
    print(f"  LP bound:  ${result.lp_relaxation:,.2f}")
    print(f"  Gap:       {result.gap*100:.2f}%")
    print(f"  Nodes:     {result.nodes}")
    print(f"  Wall time: {elapsed:.2f}s")

    if result.x is not None:
        print_schedule(result, T)

        # Verify feasibility
        x = result.x
        for t in range(T):
            total_out = sum(x[t * N + i] for i in range(N))
            if total_out < demand[t] - 1.0:
                print(f"  [WARNING] Demand not met at t={t}: {total_out:.1f} < {demand[t]}")
            else:
                pass  # OK


if __name__ == "__main__":
    import sys
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_demo(T)
