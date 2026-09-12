# ACTIVE_SPEC.md — Sovereign Optimization Solver Core (SIH)

_Single source of truth. Written by: SkilledAgent kickoff. Approved by: user (AryanMotiani)._

---

## 1. Problem Statement

India's critical sectors (refining, power, logistics, manufacturing) depend on foreign commercial solvers (Gurobi, CPLEX, FICO Xpress) for mathematical optimization. This creates national strategic risk through:
- High recurring license costs
- Restricted algorithm visibility and adaptability
- No Indian sovereign control over optimization infrastructure

Open-source alternatives (HiGHS, CBC, SCIP, GLPK) exist but are not built, validated, or tuned for Indian industrial use cases and still lag on some problem classes.

**This project builds a sovereign solver core from scratch — not wrapping any existing solver library.**

---

## 2. Solution Overview

A Python-native mathematical optimization engine supporting:
- **Phase 1:** LP (via revised simplex + IPM + PDHG), MILP (B&B + B&C), QP (ADMM/OSQP-style)
- **Phase 2 roadmap:** MIQP, NLP, MINLP
- **Interface:** Python API + CLI (`solver/`) + MPS/LP file format parsing
- **Benchmarks:** Netlib (LP), MIPLIB (MILP), Mittelmann reference sets; compared to HiGHS

---

## 3. Target Users / Judges

- SIH technical judges evaluating numerical correctness, algorithm sophistication, and benchmark performance
- Future: Indian industrial optimization teams (refinery, power dispatch, logistics)

---

## 4. User Stories

| ID | Story | Acceptance Criteria |
|---|---|---|
| US-01 | As a judge, I can feed an MPS file to the solver CLI and get a solution | `python -m solver solve instance.mps` prints status, objective, and solution vector |
| US-02 | As a judge, I can see benchmark results vs HiGHS on Netlib LP instances | CSV + performance profile plot in `benchmarks/results_lp.csv` |
| US-03 | As a judge, I can see benchmark results on MIPLIB MILP instances | CSV + plot in `benchmarks/results_milp.csv` |
| US-04 | As a judge, I can run the Indian industrial demo models (crude blending, refinery scheduling) | Both models solve correctly, output matches HiGHS on same model |
| US-05 | As a judge, I can see clear comparison tables: presolve ON/OFF, branching rules A/B/C, warm-start vs cold-start | Three comparison tables in the benchmark report |
| US-06 | As a developer, I can import the solver as a Python library | `from solver import Solver; s = Solver(); s.add_mps('f.mps'); s.solve()` |
| US-07 | As a judge, I can see evidence of GPU-accelerated PDHG (or an honest explanation of why it wasn't faster) | PDHG results table or documented go/no-go decision in report |

---

## 5. Architecture

```
sovereign-solver/
├── solver/                      # The actual solver — no external solver deps
│   ├── __init__.py              # Public API: Solver class
│   ├── __main__.py              # CLI entry point: python -m solver solve <file>
│   ├── config.py                # All named algorithm constants (tolerances, thresholds)
│   ├── problem.py               # Problem dataclass (c, A_ub, b_ub, A_eq, b_eq, lb, ub, integer_mask, P_qp)
│   ├── io/
│   │   └── mps_reader.py        # MPS format parser → Problem
│   ├── lp/
│   │   ├── simplex_dense.py     # Dense tableau simplex (correctness scaffold, Phase 1A only)
│   │   ├── simplex_revised.py   # Primal + Dual revised simplex with sparse LU
│   │   ├── interior_point.py    # Mehrotra predictor-corrector IPM
│   │   └── pdhg.py              # Restarted PDHG (CPU NumPy; CuPy port conditional)
│   ├── presolve/
│   │   ├── presolve.py          # Orchestrator: runs all reductions in order
│   │   └── postsolve.py         # Maps presolved solution → original space
│   ├── milp/
│   │   ├── branch_and_bound.py  # B&B tree: node selection, branching, LP bounding
│   │   ├── branching.py         # Most-fractional, pseudocost, reliability, on-the-fly learned ranker
│   │   ├── cuts.py              # MIR, Cover, Clique cut generators + cut selection scoring
│   │   └── heuristics.py        # Rounding/diving, Feasibility Pump, RINS, Local Branching
│   ├── qp/
│   │   └── admm.py              # OSQP-style ADMM for convex QP
│   └── utils/
│       ├── feasibility.py       # Shared is_feasible(problem, x) checker
│       └── sparse_lu.py         # Sparse LU factorization + Forrest-Tomlin update
├── tests/                       # pytest: mirrors Part B test spec
│   ├── test_env.py
│   ├── test_mps_reader.py       # B.1
│   ├── test_simplex.py          # B.2
│   ├── test_ipm.py              # B.3
│   ├── test_pdhg.py             # B.4
│   ├── test_presolve.py         # B.5
│   ├── test_milp.py             # B.6
│   ├── test_branching.py        # B.7
│   ├── test_cuts.py             # B.8
│   ├── test_heuristics.py       # B.9
│   ├── test_qp.py               # B.10
│   └── test_demos.py            # B.11
├── benchmarks/
│   ├── run_lp_bench.py          # Runs all Netlib instances: our solver vs HiGHS
│   ├── run_milp_bench.py        # Runs MIPLIB instances: our solver vs HiGHS
│   ├── run_comparison_tables.py # Presolve ON/OFF, branching A/B/C, warm/cold-start
│   ├── plot_performance.py      # Performance profile plots
│   └── results/                 # CSVs + PNGs generated by benchmark runs
├── instances/
│   ├── netlib/                  # Downloaded LP instances (.mps)
│   ├── miplib/                  # Downloaded MILP instances (.mps)
│   └── demo/                    # Hand-built Indian industrial demo models
│       ├── crude_blending.mps
│       └── refinery_scheduling.mps
├── report/
│   └── benchmark_report.md      # Final report with all tables and plots embedded
├── .agents/                     # SkilledAgent workspace (this file lives here)
├── PROBLEM_STATEMENT.md
├── README.md
└── requirements.txt
```

---

## 6. Module Test Specifications (Part B — verification gates)

### T-01: MPS Reader (B.1)
- Hand-crafted 3-var, 2-constraint MPS file → exact array match
- Parse 5 Netlib instances; `n_vars`, `n_constraints`, `c` vector match `highspy` reader (tol 1e-9)
- Gate: zero exceptions on RANGES, free rows, INTORG/INTEND markers

### T-02: Dense Simplex Scaffold (B.2 toy only)
- 2-variable LP: `max 3x+5y s.t. x≤4, 2y≤12, 3x+2y≤18` → optimal `x=2, y=6, obj=36` (tol 1e-6)
- Beale cycling example with Bland's rule → terminates, correct optimum

### T-03: Revised Sparse Simplex (B.2 full)
- All B.2 toy checks pass
- 90-instance Netlib cross-check: `|our_obj - highs_obj| / max(1, |highs_obj|) < 1e-6` on every solved instance
- Correct INFEASIBLE/UNBOUNDED status on hand-constructed cases

### T-04: Interior-Point Method (B.3)
- Netlib cross-check (same as T-03, independent code path)
- Duality gap strictly decreasing per iteration (assert monotone decrease on average)
- Cross-agreement with revised simplex on all shared instances (tol 1e-6)

### T-05: PDHG (B.4)
- 2-variable toy LP → converges to correct optimum (tol 1e-6)
- MIPLIB LP relaxation subset: convergence within 1e-6 of HiGHS
- Infeasibility/unboundedness detection: iterates diverge in certified direction (not silent loop-out)

### T-06: Presolve (B.5)
- **Non-negotiable:** for every test instance, `original_obj == presolved_then_postsolved_obj` (tol 1e-6)
- Reduction counts non-negative, reduced problem never larger than original
- Presolve ON vs OFF node-count table (no regression on aggregate)

### T-07: Basic B&B (B.6)
- 5-item knapsack toy: exact known optimum, all returned solutions pass `is_feasible()`
- Brute-force cross-check: all ≤20-binary-variable instances, 100% match
- MIPLIB subset: objective match (tol 1e-6) on instances both solvers prove optimal; gap reported otherwise
- Parent/child LP bound monotonicity assertion: never fires during any test run

### T-08: Branching Rules A/B/C (B.7)
- A/B/C comparison table: most-fractional vs reliability pseudocost vs on-the-fly ranker
- Honest reporting: include instances where simpler rule wins
- No correctness regression from T-07

### T-09: Warm-Started Dual Simplex (B.7 extension)
- T-07 correctness unaffected after wiring warm-start
- Wall-clock improvement table: warm-start vs cold-start per-instance

### T-10: Cutting Planes (B.8)
- **Non-negotiable:** every cut satisfies all enumerated feasible integer points on brute-force toy set
- LP-relaxation bound improves (or stays same) — never loosens after cut addition
- Node-count comparison: cuts ON vs OFF table

### T-11: Primal Heuristics (B.9)
- **Non-negotiable:** every incumbent returned by any heuristic passes `is_feasible(problem, x)`
- Time-to-first-solution improvement table: heuristics ON vs OFF

### T-12: QP ADMM (B.10)
- Toy QP: `min x²+y² s.t. x+y≥1` → `x=y=0.5, obj=0.5` (tol 1e-6)
- Cross-check vs HiGHS QP on constructed QP instances (tol 1e-4)
- Correct infeasibility/unboundedness detection

### T-13: Domain Demo Models (B.11)
- Crude blending LP: our solver == HiGHS objective (tol 1e-6)
- Refinery scheduling MILP: our solver == HiGHS objective on same model (tol 1e-6 if proven optimal; gap reported otherwise)
- Both models exercise at least 2 solver modules each (cover cuts, RINS)

---

## 7. Out of Scope (Phase 1)

- GUI / web interface
- NLP (Nonlinear Programming) engine
- MINLP engine
- Trained GNN branching (Gasse et al. — roadmap Phase 3)
- Neural Diving (Nair et al. — roadmap Phase 3)
- Multi-GPU B&B (roadmap Phase 4)
- Dantzig-Wolfe / column generation (roadmap Phase 4)
- Benders decomposition (roadmap Phase 4)
- Symmetry/orbital branching (roadmap Phase 4)

---

## 8. Performance Targets

| Module | Target |
|---|---|
| LP Netlib instances | ≥80% solved within 5× HiGHS wall-clock |
| MIPLIB instances (300s limit) | Objective match on ≥50% of instances both solvers prove optimal |
| Crude blending demo (LP) | Solves in < 30s, objective correct to 1e-6 vs HiGHS |
| Refinery scheduling demo (MILP) | Feasible solution found in < 120s, gap reported |
| QP toy problems | Correct to 1e-6 in < 1s |

---

## 9. Non-Negotiable Invariants

1. **No solver library inside `solver/`** (ADR-001)
2. **Every heuristic output passes `is_feasible()`** (T-11)
3. **Presolve: `original_obj == postsolved_obj`** (T-06)
4. **Every cut is valid on all enumerated feasible integer points** (T-10)
5. **B&B parent/child LP bound monotonicity** (T-07)
6. **Every benchmark number traces to a CSV row with timestamp** (ADR-005)
