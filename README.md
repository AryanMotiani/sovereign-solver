# Sovereign Solver

An industrial-grade LP/MILP/QP optimization engine built from scratch — no dependency on any existing open-source solver (no HiGHS, CBC, GLPK, SCIP). Developed as India's indigenous answer to commercial solvers (CPLEX, Gurobi, Xpress) for use in refining, petrochemicals, power, logistics, and manufacturing planning.

---

## Architecture

```
solver/
├── config.py                   # All algorithm constants (single source of truth)
├── problem.py                  # Problem dataclass (LP/MILP/QP standard form)
├── lp/
│   ├── simplex_dense.py        # Phase 1A: Dense Big-M simplex
│   ├── simplex_revised.py      # Phase 1B: Sparse revised simplex + LU
│   ├── interior_point.py       # Phase 1B: Mehrotra predictor-corrector IPM
│   └── pdhg.py                 # Phase 1C: Chambolle-Pock PDHG (first-order)
├── presolve/
│   ├── presolve.py             # Phase 2A: LP presolve (5 reductions)
│   └── postsolve.py            # Phase 2A: Solution reconstruction (LIFO stack)
├── milp/
│   ├── branch_and_bound.py     # Phase 2B: Best-first B&B MILP engine
│   ├── branching.py            # Phase 2C: Pseudocost + strong branching
│   ├── cuts.py                 # Phase 3A: MIR, cover, clique cuts
│   └── heuristics.py           # Phase 3B: Rounding, diving, Feas. Pump, RINS
├── qp/
│   └── admm.py                 # Phase 4A: ADMM QP solver (Boyd et al. 2010)
└── utils/
    └── feasibility.py          # Constraint/integrality checker

demos/
├── crude_blend_lp.py           # Phase 4B: Crude oil blending LP
└── refinery_scheduling_milp.py # Phase 4B: Refinery unit scheduling MILP

benchmarks/
├── run_benchmarks.py           # Phase 5: Full benchmark suite
└── results/                    # CSV outputs from benchmark runs

tests/                          # 130+ tests, all green
```

---

## Implemented Algorithms

### LP Solvers
| Algorithm | Module | Gate |
|---|---|---|
| Dense Big-M Simplex | `lp/simplex_dense.py` | T-01 ✅ |
| Sparse Revised Simplex (LU) | `lp/simplex_revised.py` | T-02/03 ✅ |
| Mehrotra Predictor-Corrector IPM | `lp/interior_point.py` | T-04 ✅ |
| Chambolle-Pock PDHG | `lp/pdhg.py` | T-05 ✅ |

### MILP Engine
| Component | Module | Gate |
|---|---|---|
| LP Presolve (5 reductions) + Postsolve | `presolve/` | T-06 ✅ |
| Best-first Branch-and-Bound | `milp/branch_and_bound.py` | T-07 ✅ |
| Pseudocost + Strong Branching | `milp/branching.py` | T-08 ✅ |
| MIR + Cover + Clique cuts | `milp/cuts.py` | T-10 ✅ |
| Rounding / Diving / Feas. Pump / RINS | `milp/heuristics.py` | T-11 ✅ |

### QP Solver
| Algorithm | Module | Gate |
|---|---|---|
| ADMM (saddle-point KKT system) | `qp/admm.py` | T-12 ✅ |

### Domain Demos
| Demo | Model | Result |
|---|---|---|
| Crude Oil Blending | LP, 3 crudes, 3 constraints | Optimal in 3 pivots |
| Refinery Unit Scheduling | MILP, 30 vars, 38 rows, 10 binary | Optimal in 11 B&B nodes |

---

## Benchmark Results (Phase 5)

Run `python -m benchmarks.run_benchmarks` to reproduce.

Key numbers from the standard benchmark suite:

| Problem | Simplex | IPM | PDHG |
|---|---|---|---|
| 2-var LP | 0.00s / 2 iters | 0.00s / 4 iters | 1.19s (iter limit) |
| 50-var LP (n=50, m=30) | 0.015s / 19 iters | 0.047s / 9 iters | 3.95s (iter limit) |

| MILP Problem | Nodes | Time |
|---|---|---|
| 8-var knapsack | 51 | 0.17s |
| 12-var knapsack | 27 | 0.17s |
| 16-var knapsack | 59 | 0.30s |
| Refinery scheduling (30 vars, 10 binary) | 11 | 0.28s |

---

## Running Tests

```bash
pytest tests/ -q
# Expected: 130+ passed, <5 skipped, 0 failed
```

## Running Demos

```bash
python -m demos.crude_blend_lp
python -m demos.refinery_scheduling_milp
```

## Running Benchmarks

```bash
python -m benchmarks.run_benchmarks
# Results saved to benchmarks/results/benchmark_results.csv
```

---

## Key Design Decisions

- **No existing solver dependency**: built entirely from mathematical foundations
- **Single config file**: all algorithm constants in `solver/config.py`
- **Verification-first**: no module marked complete without a passing test gate
- **Traced benchmarks**: every performance number is reproducible via CSV
- **Standard form**: all problem types normalize to `min cᵀx, Ax ≤ b, x ≥ 0` internally
- **Upper bounds**: `_build_standard_form` converts finite `ub[j]` to explicit `x_j ≤ ub_j` rows

---

## Phases Roadmap

- [x] Phase 1A: Dense Simplex
- [x] Phase 1B: Revised Simplex + IPM  
- [x] Phase 1C: PDHG (first-order)
- [x] Phase 2A: Presolve/Postsolve
- [x] Phase 2B: Branch-and-Bound MILP
- [x] Phase 2C: Advanced Branching (pseudocost, strong branching)
- [x] Phase 3A: Cutting Planes (MIR, cover, clique)
- [x] Phase 3B: Primal Heuristics (rounding, diving, FP, RINS)
- [x] Phase 4A: QP via ADMM
- [x] Phase 4B: Domain Demos (crude blend, refinery scheduling)
- [x] Phase 5: Benchmark Suite
- [ ] Phase 5+: GPU PDHG, LP warmstarting, rolling horizon, report

---

*Built for Smart India Hackathon — Problem Statement: Sovereign LP/MILP Solver*
