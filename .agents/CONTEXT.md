# CONTEXT.md — Sovereign Solver (SIH)

_Shared memory: standards, domain language, architectural decisions, ADRs._

---

## Project Identity

- **Name:** Sovereign Optimization Solver Core
- **Competition:** Smart India Hackathon (SIH)
- **Repo:** `AryanMotiani/sovereign-solver` (private)
- **Goal:** A from-scratch mathematical optimization engine for LP, MILP, and QP — NOT built on any existing solver (no GLPK, HiGHS, SCIP, CBC as dependencies inside the solver itself).
- **Baseline comparison:** `highspy` (HiGHS Python bindings) used ONLY in the benchmark harness, never imported by the solver.

---

## Ubiquitous Language (Domain Glossary)

| Term | Definition |
|---|---|
| **LP** | Linear Program: `min cᵀx s.t. Ax ≤ b, lb ≤ x ≤ ub` |
| **MILP** | Mixed-Integer LP — some variables `x_i ∈ ℤ` |
| **QP** | Quadratic Program — objective has a quadratic term `½xᵀPx + qᵀx` |
| **Simplex** | Pivoting LP algorithm that walks basic feasible solution vertices |
| **Revised Simplex** | Sparse form: maintains basis matrix `B`, not full tableau |
| **IPM / Interior-Point** | Barrier method traversing the interior of the feasible set; Mehrotra predictor-corrector |
| **PDHG** | Primal-Dual Hybrid Gradient — first-order method, GPU-friendly |
| **B&B** | Branch-and-Bound — MILP tree search; LP relaxations bound each node |
| **B&C** | Branch-and-Cut — B&B + cutting planes added at nodes |
| **Presolve** | Problem reduction before solving (remove redundancies, tighten bounds) |
| **Postsolve** | Map presolved solution back to original variable space |
| **MIR Cut** | Mixed-Integer Rounding cut — derived from a single aggregated constraint row |
| **Cover Cut** | Valid inequality for knapsack-type binary constraints |
| **Clique Cut** | Valid inequality from a clique in the variable conflict graph |
| **Feasibility Pump (FP)** | Heuristic: round LP → repair roundings via distance-LP; iterate |
| **RINS** | Relaxation Induced Neighborhood Search — fix vars agreeing between LP relaxation and incumbent; solve sub-MIP |
| **Local Branching** | Fix Hamming-distance ≤ k constraint around incumbent; solve sub-MIP |
| **Pseudocost** | Historical average of objective improvement per unit of branching fractionality for a variable |
| **Reliability Branching** | Use strong branching until pseudocosts are "reliable" (N updates), then switch to pseudocost |
| **Strong Branching** | Oracle: solve both child LPs, pick variable giving best bound improvement |
| **MPS Format** | Standard optimization problem file format (Netlib, MIPLIB instances) |
| **MIPLIB** | Mixed-Integer Programming Library — standard MILP benchmark instances |
| **Netlib** | Standard LP benchmark library |
| **Optimality Gap** | `(incumbent_obj - best_bound) / |best_bound|` — how close we are to proven optimum |
| **Duality Gap** | IPM-specific: `primal_obj - dual_obj`; must decrease monotonically per iteration |

---

## Architectural Decision Records (ADRs)

### ADR-001: Build from scratch — no solver library dependencies inside the solver

**Decision:** The solver core (`solver/` directory) must not import or call into CPLEX, Gurobi, Xpress, HiGHS, SCIP, CBC, GLPK, or any other solver at runtime. These may only appear in `benchmarks/` as comparison baselines.

**Rationale:** SIH problem statement explicitly requires this. Sovereignty is the core value proposition.

**Consequence:** All linear algebra, factorization, and pivoting must be implemented from scratch in NumPy/SciPy (dense baseline first, sparse LU upgrade later).

---

### ADR-002: Language = Python (NumPy core), with Rust/C++ extension as stretch goal

**Decision:** Phase 1–3 prototype in Python/NumPy. If performance is insufficient for large benchmarks, write hot paths (LU factorization, sparse matrix-vector products) as C extensions via ctypes or a small Rust library via PyO3.

**Rationale:** Python enables fastest iteration for a hackathon. The algorithmic logic is the differentiator, not the language.

---

### ADR-003: LP engine is dual simplex (not primal) inside B&B

**Decision:** Within the B&B loop, the LP at each node is solved with dual simplex re-optimization (warm-starting from the parent's optimal basis with one bound change), not primal simplex from scratch.

**Rationale:** Dual feasibility is preserved after a bound-tightening branch; primal feasibility is not. Dual simplex + warm start is 10-100x faster per node than cold-start primal.

---

### ADR-004: MIR cuts as primary cut family (not Gomory)

**Decision:** Implement MIR cuts first. Gomory cuts second (optional). Cover and clique cuts for structured instances.

**Rationale:** MIR cuts have a cleaner derivation (single constraint aggregation + rounding, no tableau-row bookkeeping) and are theoretically equivalent to Gomory mixed-integer cuts. Easier to implement correctly from scratch.

---

### ADR-005: Verification policy — no hallucinated completions

**Decision:** Every module is marked complete ONLY when its Part-B test suite (defined in ACTIVE_SPEC.md) passes. Every benchmark number must trace to a row in `benchmarks/*.csv`.

---

### ADR-006: GPU PDHG is always built CPU-first; GPU port is conditional

**Decision:** PDHG (first-order method) is implemented in NumPy first. CuPy GPU port only if ≥2x wall-clock speedup is measured over CPU simplex on large/sparse instances at benchmark time.

**Rationale:** GPU advantage is instance-size-dependent. Don't burn time porting before measuring.

---

## Standards

- **Testing framework:** `pytest`
- **Benchmark output:** CSV files in `benchmarks/*.csv` with columns: `instance, solver, status, objective, gap, wall_clock_s, nodes (MILP), iterations (LP)`
- **Float tolerance:** Correctness checks at `1e-6` relative; PDHG/ADMM at `1e-4`
- **Code style:** PEP-8, type hints on all public functions
- **No magic numbers:** All algorithm parameters (pivot tolerance, reliability threshold, cut scoring weights) are named constants in `solver/config.py`
- **Every heuristic output validated by `is_feasible(problem, x)`** before being accepted as an incumbent