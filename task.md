# task.md — Sovereign Solver Build Tracker

> **HOW TO USE THIS FILE:**
> - Update this file as you complete work.
> - `[ ]` = not started, `[/]` = in progress, `[x]` = done (gate verified)
> - An item is only `[x]` when its pytest gate is GREEN, never before.
> - Read `master_implementation_plan.md` for full ticket details before starting any item.

---

## PHASE 1A — Foundation

- [x] **1A-01** Project scaffold, `Problem` dataclass, `is_feasible()`, `config.py`, CLI stub, `tests/test_env.py`
  - Gate: `pytest tests/test_env.py` → all pass ✅
- [x] **1A-02** MPS Reader (`solver/io/mps_reader.py`)
  - Gate: T-01 → `pytest tests/test_mps_reader.py` → all pass ✅ (31 passed, afiro skipped — no internet)
- [x] **1A-03** Dense tableau simplex scaffold (`solver/lp/simplex_dense.py`)
  - Gate: T-02 → `pytest tests/test_simplex.py::test_dense_simplex_*` → all pass ✅

---

## PHASE 1B — LP Engine Core

- [ ] **1B-01** Revised primal simplex with dense LU → then sparse LU upgrade (`solver/lp/simplex_revised.py`)
  - Gate: T-03 partial → `pytest tests/test_simplex.py::test_revised_toy` → pass
- [ ] **1B-02** Dual simplex + warm-start API (`solver/lp/simplex_revised.py`)
  - Gate: T-03 full → Netlib 90-instance cross-check → `pytest tests/test_simplex.py` → all pass
- [ ] **1B-03** Mehrotra IPM (`solver/lp/interior_point.py`)
  - Gate: T-04 → `pytest tests/test_ipm.py` → all pass
- [ ] **1B-04** Sparse LU module (`solver/utils/sparse_lu.py`)
  - Gate: T-03 regression-free after swap to sparse LU

---

## PHASE 1C — PDHG (First-Order LP)

- [ ] **1C-01** CPU PDHG, restarted (`solver/lp/pdhg.py`)
  - Gate: T-05 → `pytest tests/test_pdhg.py` → all pass
- [ ] **1C-02** GPU Go/No-Go decision (documented in `benchmarks/results/gpu_decision.md`)
- [ ] **1C-02b** GPU PDHG port (conditional on GO) (`solver/lp/pdhg_gpu.py`)

---

## PHASE 2A — Presolve

- [ ] **2A-01** Presolve reductions 1–7 (`solver/presolve/presolve.py`)
  - Gate: T-06 partial → unit tests per reduction type
- [ ] **2A-02** Postsolve (`solver/presolve/postsolve.py`)
  - Gate: T-06 full → `pytest tests/test_presolve.py` → all pass (zero obj mismatches)

---

## PHASE 2B — Basic B&B

- [ ] **2B-01** B&B tree + node data structure (`solver/milp/branch_and_bound.py`)
  - Gate: tree structure unit tests pass
- [ ] **2B-02** LP bounding + branching loop + most-fractional branching
  - Gate: T-07 → `pytest tests/test_milp.py` → all pass (including brute-force toy cross-check)

---

## PHASE 2C — Advanced Branching

- [ ] **2C-01** Reliability pseudocost branching + strong branching (`solver/milp/branching.py`)
  - Gate: T-08 partial
- [ ] **2C-02** On-the-fly learned ranker (ridge regression, Khalil 2016 features)
  - Gate: T-08 full → A/B/C comparison table produced → `pytest tests/test_branching.py` → all pass

---

## PHASE 2D — Warm-Start Dual Simplex

- [ ] **2D-01** Wire dual simplex warm-start into B&B node processing
  - Gate: T-09 → T-07 correctness re-confirmed + wall-clock improvement table

---

## PHASE 3A — Cutting Planes

- [ ] **3A-01** MIR cut generator + cut selection scoring (`solver/milp/cuts.py`)
  - Gate: T-10 partial → validity on toy instances
- [ ] **3A-02** Cover cuts
  - Gate: T-10 partial
- [ ] **3A-03** Clique cuts + conflict graph
  - Gate: T-10 full → `pytest tests/test_cuts.py` → all pass + node-count comparison table

---

## PHASE 3B — Primal Heuristics

- [ ] **3B-01** Rounding + diving heuristic (`solver/milp/heuristics.py`)
  - Gate: T-11 partial → `is_feasible` on all outputs
- [ ] **3B-02** Feasibility Pump (+ Achterberg-Berthold improvement)
  - Gate: T-11 partial
- [ ] **3B-03** RINS
  - Gate: T-11 partial
- [ ] **3B-04** Local Branching
  - Gate: T-11 full → `pytest tests/test_heuristics.py` → all pass + time-to-first-solution table

---

## PHASE 4A — QP ADMM

- [ ] **4A-01** OSQP-style ADMM solver (`solver/qp/admm.py`)
  - Gate: T-12 → `pytest tests/test_qp.py` → all pass

---

## PHASE 4B — Demo Models

- [ ] **4B-01** Crude blending LP (`instances/demo/crude_blending.mps` + harness)
  - Gate: T-13 partial → solver == HiGHS obj + sanity check
- [ ] **4B-02** Refinery scheduling MILP (`instances/demo/refinery_scheduling.mps` + harness)
  - Gate: T-13 full → `pytest tests/test_demos.py` → all pass

---

## PHASE 5 — Benchmarks & Report

- [ ] **5-01** LP benchmark harness (`benchmarks/run_lp_bench.py`)
  - Gate: `benchmarks/results/lp_results.csv` exists with all Netlib instances
- [ ] **5-02** MILP benchmark harness (`benchmarks/run_milp_bench.py`)
  - Gate: `benchmarks/results/milp_results.csv` exists with MIPLIB subset
- [ ] **5-03** Comparison tables (presolve ON/OFF, branching A/B/C, warm/cold-start)
  - Gate: three CSVs in `benchmarks/results/`
- [ ] **5-04** Performance profile plots (`benchmarks/plot_performance.py`)
  - Gate: two PNG files in `benchmarks/results/`
- [ ] **5-05** Final benchmark report (`report/benchmark_report.md`)
  - Gate: every number traces to a CSV row; no hand-typed figures

---

## Final Commit Checklist

- [ ] `pytest` → 100% pass
- [ ] `git add -A && git commit -m "final: complete sovereign solver prototype" && git push`
- [ ] All benchmark CSVs timestamped and committed
- [ ] Report committed

---

**Last updated:** 2026-09-12 | **Status:** Phase 1A — NOT STARTED
