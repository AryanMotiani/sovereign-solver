# Sovereign Solver

An industrial-grade **LP/MILP/QP** optimization engine built **from scratch** — no dependency on any existing open-source solver (no HiGHS, CBC, GLPK, SCIP). Developed as India's indigenous answer to commercial solvers (CPLEX, Gurobi, Xpress) for use in refining, petrochemicals, power, logistics, and manufacturing planning.

> **SIH Problem**: Development of a Sovereign Mathematical Optimization Solver Core

---

## Architecture

```
solver/
├── config.py                   # All algorithm constants (single source of truth)
├── problem.py                  # Problem dataclass (LP/MILP/QP standard form)
├── solver.py                   # High-level API: Solver class
├── __main__.py                 # CLI: python -m solver solve <file.mps|.lp|.qps>
├── io/
│   ├── mps_reader.py           # MPS format parser
│   ├── lp_reader.py            # LP format parser (CPLEX/Gurobi style)
│   └── qps_reader.py           # QPS format parser (MPS + QUADOBJ)  [NEW]
├── lp/
│   ├── simplex_dense.py        # Dense Big-M simplex (Phase 1A)
│   ├── simplex_revised.py      # Sparse revised simplex + LU (Phase 1B)
│   ├── interior_point.py       # Mehrotra predictor-corrector IPM (Phase 1B)
│   └── pdhg.py                 # Chambolle-Pock PDHG first-order (Phase 1C)
├── presolve/
│   ├── presolve.py             # Presolve reductions (7 strategies incl. probing) [NEW: probing]
│   └── postsolve.py            # Solution reconstruction (LIFO stack)
├── milp/
│   ├── branch_and_bound.py     # Full B&C pipeline
│   ├── branching.py            # Pseudocost + reliability + strong branching
│   ├── cuts.py                 # Gomory + MIR + cover + clique cuts
│   └── heuristics.py           # Rounding, diving, Feas. Pump, RINS, Local Branching
├── qp/
│   └── admm.py                 # ADMM QP solver
└── utils/
    ├── sparse_lu.py            # SparseLU wrapper (COLAMD ordering)
    └── feasibility.py          # Constraint/integrality checker

demos/
├── crude_blend_lp.py           # Crude oil blending LP
├── refinery_scheduling_milp.py # Refinery unit scheduling MILP
├── power_economic_dispatch.py  # Power system economic dispatch + unit commitment
├── supply_chain_milp.py        # Supply chain network design MILP  [NEW]
└── transportation_lp.py        # Transportation problem LP (India rail freight)  [NEW]

benchmarks/
├── run_benchmarks.py           # Internal benchmark suite
├── netlib_benchmark.py         # Netlib LP vs HiGHS comparison
├── miplib_benchmark.py         # MIPLIB-style MILP benchmark  [NEW]
└── results/                    # CSV outputs

tests/                          # 169+ tests, all green
```

---

## Implemented Algorithms (Problem Statement §2 Coverage)

### LP Solvers
| Algorithm | Module | Status |
|-----------|--------|--------|
| Dense Big-M Simplex (Phase 1/2) | `lp/simplex_dense.py` | ✅ |
| Sparse Revised Simplex + LU (Primal + Dual) | `lp/simplex_revised.py` | ✅ |
| Mehrotra Predictor-Corrector IPM | `lp/interior_point.py` | ✅ |
| Chambolle-Pock PDHG (first-order) | `lp/pdhg.py` | ✅ |
| Variable upper bounds (IPM + PDHG) | All LP solvers | ✅ |
| AMD/COLAMD reordering | `utils/sparse_lu.py` (via `splu`) | ✅ |
| Bland's rule anti-cycling | `simplex_revised.py` | ✅ |

### MILP Engine (Full Branch-and-Cut Pipeline)
| Component | Module | Status |
|-----------|--------|--------|
| Presolve (5 reductions) + Postsolve | `presolve/` | ✅ |
| Best-first Branch-and-Bound tree | `milp/branch_and_bound.py` | ✅ |
| Pseudocost + Reliability + Strong Branching | `milp/branching.py` | ✅ |
| **Gomory fractional cuts** (new) | `milp/cuts.py` | ✅ |
| MIR cuts | `milp/cuts.py` | ✅ |
| Cover cuts | `milp/cuts.py` | ✅ |
| Clique cuts | `milp/cuts.py` | ✅ |
| Feasibility Pump | `milp/heuristics.py` | ✅ |
| RINS | `milp/heuristics.py` | ✅ |
| Diving heuristics | `milp/heuristics.py` | ✅ |
| Local Branching | `milp/heuristics.py` | ✅ |
| Presolve-space / original-space objective tracking | `branch_and_bound.py` | ✅ |

### QP Solver
| Algorithm | Module | Status |
|-----------|--------|--------|
| ADMM (saddle-point KKT) | `qp/admm.py` | ✅ |

---

## Industrial Demos (Problem Statement §3)

| Domain | Demo | Status |
|--------|------|--------|
| Crude oil blending | `demos/crude_blend_lp.py` | LP | ✅ |
| Refinery scheduling | `demos/refinery_scheduling_milp.py` | MILP | ✅ |
| Power economic dispatch + unit commitment | `demos/power_economic_dispatch.py` | MILP | ✅ |
| **Supply chain network design** | `demos/supply_chain_milp.py` | MILP | ✅ (new) |
| **Transportation & logistics** | `demos/transportation_lp.py` | LP | ✅ (new) |

### File Format Support
| Format | Module | Status |
|--------|--------|--------|
| MPS | `io/mps_reader.py` | ✅ |
| LP (CPLEX/Gurobi style) | `io/lp_reader.py` | ✅ |
| QPS (MPS + QUADOBJ) | `io/qps_reader.py` | ✅ (new) |

---

## Benchmark Results

### Netlib LP Comparison vs HiGHS

Run `py -3 -m benchmarks.netlib_benchmark` (downloads MPS files from HiGHS GitHub):

| Instance | Sovereign Simplex Error | IPM Error | HiGHS |
|----------|------------------------|-----------|-------|
| AFIRO (n=32) | **1.2e-16** (machine precision) | 2.2e-6 | -464.753 |
| ADLITTLE (n=97) | **3.9e-16** (machine precision) | 4.6e-6 | 225495 |
| AVGAS (n=8) | **0.0** (exact match) | 8.4e-7 | -7.750 |

> Sovereign Simplex matches HiGHS to **machine precision** on standard LP instances.

### Internal Benchmark Suite

| Problem | Simplex | IPM | PDHG |
|---------|---------|-----|------|
| 2-var LP | 0.00s / 2 iters | 0.00s / 4 iters | converges |
| 50-var LP | 0.015s / 19 iters | 0.047s / 9 iters | — |
| Klee-Minty n=5 | solves correctly | — | — |
| Klee-Minty n=8 | solves correctly | — | — |

| MILP Problem | Nodes | Time |
|--------------|-------|------|
| 0-1 Knapsack | 1-27 nodes | <0.3s |
| Refinery scheduling (30 vars, 10 binary) | 11 | 0.28s |
| Power dispatch T=4 (42 binary, 24 continuous) | 1 | 0.49s |

---

## Usage

### Python API

```python
from solver import Solver, read_mps, read_lp

# Load and solve from MPS
s = Solver()
s.read_mps("instances/netlib/afiro.mps")
result = s.solve()
print(result.status, result.objective)

# Load from LP format
s2 = Solver()
s2.read_lp("my_problem.lp")
result2 = s2.solve(method="milp")
```

### CLI

```bash
# Solve MPS file (auto-detects LP/MILP)
python -m solver solve instances/netlib/afiro.mps

# Solve LP file
python -m solver solve my_problem.lp --method simplex

# Specify method
python -m solver solve problem.mps --method ipm

# Problem info
python -m solver info problem.mps
```

### Running Tests

```bash
pytest tests/ -q
# Expected: 169+ passed, <5 skipped, 0 failed
```

### Running Demos

```bash
python -m demos.crude_blend_lp
python -m demos.refinery_scheduling_milp
python -m demos.power_economic_dispatch        # Power system unit commitment
python -m demos.supply_chain_milp              # Supply chain network design  [NEW]
python -m demos.transportation_lp              # India rail freight transport  [NEW]
```

### Running Benchmarks

```bash
python -m benchmarks.run_benchmarks            # Internal suite
py -3 -m benchmarks.netlib_benchmark           # Netlib vs HiGHS
py -3 -m benchmarks.miplib_benchmark --only-builtin  # MIPLIB-style (no download)  [NEW]
py -3 -m benchmarks.miplib_benchmark           # With stein27/stein45 download
```

---

## Key Design Decisions

- **No existing solver dependency**: built entirely from mathematical foundations
- **Single config file**: all algorithm constants in `solver/config.py`
- **Dual incumbent tracking**: `incumbent_obj` in original problem space (for correct reporting after presolve), `_prune_obj` in presolved space (for efficient B&B pruning)
- **Full pipeline integration**: presolve → root LP → heuristics → cuts → B&B tree → postsolve
- **COLAMD ordering**: scipy's `splu` uses COLAMD column reordering for numerical stability
- **Standard form**: all problem types normalize to `min cᵀx, Ax ≤ b, lb ≤ x ≤ ub` internally
- **Verification-first**: no module marked complete without a passing test gate

---

## What Requires Human Action (Out of Scope for Automated Build)

1. **Multi-core parallel B&B** — Python GIL limits; needs `multiprocessing` + shared memory
2. **C++/Rust bindings** — needs pybind11/maturin build environment
3. **GPU acceleration** — needs CUDA-capable GPU + CuPy
4. **Full MIPLIB benchmark** — 200+ instances, ~1GB download from miplib.zib.de
5. **Mittelmann benchmark** — large instances at plato.asu.edu/ftp/lptestset/

---

## Roadmap

- [x] Phase 1A: Dense Simplex
- [x] Phase 1B: Revised Simplex + IPM
- [x] Phase 1C: PDHG (first-order)
- [x] Phase 2A: Presolve/Postsolve
- [x] Phase 2B: Branch-and-Bound MILP
- [x] Phase 2C: Advanced Branching (pseudocost, strong, reliability)
- [x] Phase 3A: Cutting Planes (Gomory, MIR, cover, clique)
- [x] Phase 3B: Primal Heuristics (rounding, diving, FP, RINS, Local Branching)
- [x] Phase 4A: QP via ADMM
- [x] Phase 4B: Domain Demos (crude blend, refinery, **power dispatch**)
- [x] Phase 5: Netlib benchmark vs HiGHS
- [x] Phase 5+: LP format parser, CLI LP support, advanced tests
- [ ] Phase 6: GPU PDHG, C++ bindings, MIPLIB benchmark

---

*Built for Smart India Hackathon — Problem Statement: Sovereign LP/MILP Solver*
