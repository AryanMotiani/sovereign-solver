# Sovereign Solver — Master Implementation Plan

> **SkilledAgent Kickoff Output** | Phases 0-7 complete | Ready for Phase 8 (ticket execution)
>
> This document is the single canonical plan. Any agent or human picking this up can determine
> exactly where work stands, what to do next, and how to verify it is done correctly.
> **Read `.agents/CONTEXT.md` and `.agents/ACTIVE_SPEC.md` before starting any ticket.**

---

## Project Goal

Build a sovereign LP/MILP/QP optimization solver core from scratch in Python (NumPy), benchmark it against HiGHS on Netlib and MIPLIB instances, and demonstrate it on Indian industrial demo models (crude blending LP + refinery scheduling MILP).

**Critical constraint:** The solver core (`solver/`) must NEVER import HiGHS, SCIP, CBC, GLPK, Gurobi, CPLEX, or any other solver. HiGHS (`highspy`) appears ONLY in `benchmarks/` as a comparison baseline.

---

## Phase Map

```
PHASE 1A — Foundation (Environment + MPS Reader + Dense Simplex scaffold)
PHASE 1B — LP Engine Core (Revised Simplex + IPM)
PHASE 1C — LP Novelty (PDHG first-order method, CPU then conditional GPU)
PHASE 2A — Presolve
PHASE 2B — Basic MILP (Branch-and-Bound, most-fractional branching)
PHASE 2C — Advanced Branching (Pseudocost + On-the-fly learned ranker)
PHASE 2D — Warm-Start Dual Simplex inside B&B
PHASE 3A — Cutting Planes (MIR + Cover + Clique)
PHASE 3B — Primal Heuristics (FP + RINS + Local Branching)
PHASE 4A — QP Engine (OSQP-style ADMM)
PHASE 4B — Indian Industrial Demo Models
PHASE 5  — Benchmarking & Report Generation
```

Each phase has: **tickets → implementation steps → verification gate (test IDs from ACTIVE_SPEC.md)**

An agent completing a phase MUST verify the gate before starting the next phase.

---

## PHASE 1A — Foundation

**Goal:** Working environment, MPS reader, and a throwaway dense simplex scaffold that proves LP logic is correct before investing in sparse machinery.

---

### Ticket 1A-01: Project scaffold & environment

**Files to create:**
```
solver/__init__.py
solver/__main__.py
solver/config.py
solver/problem.py
solver/utils/__init__.py
solver/utils/feasibility.py
requirements.txt
tests/__init__.py
tests/test_env.py
instances/netlib/        (directory)
instances/miplib/        (directory)
instances/demo/          (directory)
benchmarks/results/      (directory)
```

**Implementation steps:**
1. `requirements.txt`: `numpy scipy pytest matplotlib highspy`
2. `solver/problem.py`: define `Problem` dataclass:
   ```python
   @dataclass
   class Problem:
       c: np.ndarray           # objective coefficients (n,)
       A_ub: sp.csr_matrix     # inequality LHS (m_ub, n)
       b_ub: np.ndarray        # inequality RHS (m_ub,)
       A_eq: sp.csr_matrix     # equality LHS (m_eq, n)
       b_eq: np.ndarray        # equality RHS (m_eq,)
       lb: np.ndarray          # lower bounds (n,)
       ub: np.ndarray          # upper bounds (n,) — np.inf for free
       integer_mask: np.ndarray  # bool (n,) — True = integer variable
       P_qp: Optional[sp.csr_matrix] = None  # QP quadratic term (n, n), PSD
       sense: str = 'min'      # 'min' or 'max'
   ```
3. `solver/utils/feasibility.py`: implement `is_feasible(problem, x, tol=1e-6) -> bool`:
   - Check `A_ub @ x <= b_ub + tol` (element-wise)
   - Check `|A_eq @ x - b_eq| <= tol` (element-wise)
   - Check `lb - tol <= x <= ub + tol` (element-wise)
   - Check `|x[integer_mask] - round(x[integer_mask])| <= tol` (element-wise)
   - Returns True only if ALL checks pass
4. `solver/config.py`: define all named constants:
   ```python
   PIVOT_TOL = 1e-7
   FEASIBILITY_TOL = 1e-6
   OPTIMALITY_TOL = 1e-6
   PDHG_TOL = 1e-6
   ADMM_TOL = 1e-4
   RELIABILITY_THRESHOLD = 8   # Khalil/Achterberg: switch from strong to pseudocost after this many updates
   STRONG_BRANCH_CANDIDATES = 20  # max candidates evaluated by strong branching per node
   MAX_CUTS_PER_ROUND = 50
   CUT_VIOLATION_MIN = 1e-4
   ```
5. `solver/__main__.py`: minimal CLI stub: `python -m solver solve <file.mps>`
6. `solver/__init__.py`: export `Solver` class stub
7. `tests/test_env.py`:
   - assert `numpy`, `scipy`, `highspy` importable
   - solve a 1-variable LP via `highspy` directly, assert objective is correct
   - assert `is_feasible` returns True on a trivially feasible point

**Verification gate:** `pytest tests/test_env.py` → all pass

---

### Ticket 1A-02: MPS Reader

**File:** `solver/io/mps_reader.py`

**Implementation steps:**
1. Implement `read_mps(path: str) -> Problem` parsing these MPS sections in order:
   - `NAME` (optional)
   - `ROWS`: classify each row as N (objective/free), L (≤), G (≥), E (=)
   - `COLUMNS`: fill sparse coefficient matrix; detect `MARKER 'INTORG'`/`'INTEND'` for integer variables
   - `RHS`: right-hand side values
   - `RANGES`: convert range rows into two constraints
   - `BOUNDS`: handle UP, LO, FX, FR (free), MI (−∞ lower), PL (+∞ upper), BV (binary 0/1)
   - `ENDATA`
2. Return a `Problem` with scipy CSR sparse matrices for A_ub, A_eq
3. Normalise: convert G (≥) rows to L (≤) by negating; convert 'max' objective to 'min' by negating c (store original sense in `Problem.sense`)

**Verification gate (T-01):**
- Create `tests/instances/toy_3var.mps` by hand (3 variables, 2 constraints); assert parsed arrays match expected exactly
- Parse 5 Netlib instances AND parse the same files via `highspy`; assert `n_vars`, `n_constraints`, `c` match (tol 1e-9)
- Assert zero exceptions on RANGES, free-row, and INTORG/INTEND test files

---

### Ticket 1A-03: Dense Tableau Simplex (correctness scaffold)

**File:** `solver/lp/simplex_dense.py`

**Implementation steps:**
1. Implement the classic full-tableau simplex (dense numpy matrix, no sparse concern):
   - Add slack variables to convert to standard form
   - Dantzig's rule for pivot column selection (most negative reduced cost)
   - Minimum ratio test for pivot row selection
   - Bland's rule: if `iteration > 2 * n_vars`, switch to Bland's lexicographic rule (smallest index tie-breaking) to guarantee termination
   - Phase I (big-M or two-phase) for finding an initial BFS
   - Return `(status, x, obj)` where `status` ∈ `{'optimal', 'infeasible', 'unbounded'}`
2. This is intentionally simple — correctness over performance

**Verification gate (T-02):**
- 2-var LP: `max 3x+5y s.t. x≤4, 2y≤12, 3x+2y≤18` → `x=2, y=6, obj=36` (tol 1e-6)
- Beale cycling example (6-var known cycling LP) with Bland's rule → terminates within `10 * n_vars` iterations, correct optimum
- 1 infeasible LP → status `'infeasible'`
- 1 unbounded LP → status `'unbounded'`

---

## PHASE 1B — LP Engine Core

**Goal:** Production-quality LP engine: revised simplex (primal + dual) with sparse LU, and Mehrotra IPM. This is the core credibility layer.

---

### Ticket 1B-01: Revised Primal Simplex

**File:** `solver/lp/simplex_revised.py`

**Implementation steps:**
1. Represent basis `B` as index set; compute `B_matrix = A[:, basis]`
2. **Dense LU first:** use `scipy.linalg.lu_factor` / `lu_solve` to solve `B @ y = rhs` — get correctness first
3. Compute reduced costs: `rc = c_N - (B^{-T} c_B)^T A_N`
4. Bland's pivot selection (correctness mode); switch to steepest-edge Devex pricing after correctness confirmed
5. Minimum ratio test (primal ratio test)
6. Update basis, re-factorize periodically (every `k` pivots, e.g. k=50) or when numerical error accumulates
7. Detect: optimal (all rc ≥ 0), infeasible (Phase I), unbounded (no finite ratio)
8. **After dense version passes all tests:** swap dense LU for `scipy.sparse.linalg.splu` (sparse LU) as a drop-in; run same tests to confirm no regression

**Verification gate (T-03, partial):** All T-02 toy checks pass via this revised implementation.

---

### Ticket 1B-02: Dual Simplex

**File:** `solver/lp/simplex_revised.py` (add `dual_simplex_step` method)

**Implementation steps:**
1. Dual simplex starts from a dual-feasible but primal-infeasible basis (reduced costs all ≥ 0, but some constraints violated)
2. Pivot selection: leaving variable = most infeasible basic variable; entering variable = dual ratio test
3. Dual ratio test: `min |rc_j / a_{ij}|` over `a_{ij} > 0` columns
4. **Bound flipping** (long step): allow multiple variable bound-flips in a single dual pivot (per Huangfu & Hall 2018) — implement after basic dual simplex works
5. Warm-start API: `dual_simplex_warm_start(basis, basis_values, new_bound_change)` — used by B&B

**Verification gate (T-03, full):**
- Netlib 90-instance cross-check (both primal and dual code paths)
- Correct INFEASIBLE/UNBOUNDED on hand-constructed cases
- `pytest tests/test_simplex.py` → all pass

---

### Ticket 1B-03: Mehrotra Predictor-Corrector IPM

**File:** `solver/lp/interior_point.py`

**Algorithm (Mehrotra 1992):**
1. Convert to standard form: `min cᵀx s.t. Ax = b, x ≥ 0`
2. Initialize: `x > 0, y, s > 0` (Mehrotra's starting point heuristic)
3. Each iteration:
   a. Form and solve the **normal equations**: `(A D² Aᵀ) Δy = rhs` where `D = diag(x/s)`
   b. **Predictor step:** solve for affine-scaling direction `(Δx_aff, Δy_aff, Δs_aff)`
   c. Compute centering parameter `σ = (μ_aff / μ)³`
   d. **Corrector step:** add centering + higher-order correction to RHS
   e. Combined step: `(Δx, Δy, Δs) = predictor + corrector`
   f. Step length: `α = min(1, 0.99 * min ratio)` (Mehrotra's α rule)
   g. Update `x, y, s`
4. Convergence: `max(|primal_res|, |dual_res|, duality_gap) < tol`
5. **Normal equations solver:** dense for small problems; `scipy.sparse.linalg.spsolve` (direct sparse) for medium; sparse Cholesky (`sksparse.cholmod` if available, else `scipy`'s incomplete Cholesky as fallback) for large
6. Log duality gap per iteration for monotone-decrease assertion

**Verification gate (T-04):**
- Netlib cross-check: same subset as simplex, same 1e-6 tolerance
- Duality gap strictly decreasing per iteration (assert `gaps[i+1] < gaps[i] * 1.1` — allowing small oscillation, never sustained increase)
- IPM and simplex agree on all shared instances (tol 1e-6)
- `pytest tests/test_ipm.py` → all pass

---

### Ticket 1B-04: Sparse LU Factorization Module

**File:** `solver/utils/sparse_lu.py`

**Implementation steps:**
1. Wrap `scipy.sparse.linalg.splu` with a clean interface: `LUFactor(B_sparse)` → object with `.solve(rhs)` and `.solve_transpose(rhs)` methods
2. Implement `update(leaving_col, entering_col)`: if `scipy.sparse.linalg.splu` doesn't support rank-1 updates natively, fall back to full re-factorization every `REFACTORIZE_EVERY = 50` pivots (document this as the known limitation; Forrest-Tomlin update is the stretch goal)
3. Implement column-ordering heuristic: use `scipy.sparse.csgraph.minimum_spanning_tree` or AMD ordering via `scikit-sparse` to minimize fill-in before factorization
4. **Stretch goal (if time):** implement Forrest-Tomlin rank-1 update — track eta-matrices as a sequence and apply them in factored form

**Verification gate:** Revised simplex using this module passes all T-03 tests with no regression; `splu` factorization is confirmed faster than dense `lu_factor` on Netlib instances with n > 500 (log both timings).

---

## PHASE 1C — PDHG (First-Order LP Solver)

**Goal:** Implement a restarted PDHG LP solver (Applegate et al. 2021, PDLP) as a third, structurally independent LP algorithm. CPU-first; GPU port conditional on benchmarks.

---

### Ticket 1C-01: CPU PDHG

**File:** `solver/lp/pdhg.py`

**Algorithm (Applegate et al. arXiv:2106.04756 — PDLP/restarted PDHG):**

1. Convert to standard form: `min cᵀx s.t. Ax = b, l ≤ x ≤ u`
2. **Diagonal preconditioning:** `Dᵣ = diag(1/||Aᵢ||)`, `Dᶜ = diag(1/||Aʲ||)` — rescale so row/column norms ≈ 1
3. **PDHG iteration:**
   - Primal step: `x^{k+1} = clip(x^k - τ(Aᵀy^k + c), l, u)` where `τ` is primal step size
   - Dual step: `y^{k+1} = y^k + σ(A(2x^{k+1} - x^k) - b)` where `σ` is dual step size
   - Step sizes: `τ = σ = 1/||A||_F` initially; adaptive updates per Applegate §3
4. **Adaptive restarts:** restart (reset `y` to 0, update reference iterate) when normalized duality gap stops decreasing relative to a fixed window — use Algorithm 2 from the paper
5. **Convergence:** `max(primal_residual, dual_residual, duality_gap) / (1 + |obj|) < tol`
6. **Infeasibility/unboundedness detection:** track iterates diverging in the certified direction (per Applegate §4); flag as `INFEASIBLE` or `UNBOUNDED` rather than looping to iteration cap

**Verification gate (T-05):**
- 2-var toy LP → converges to correct optimum (tol 1e-6)
- MIPLIB LP relaxation subset: ≥70% converge within 1e-6 of HiGHS (remainder log timeout, not fail)
- Infeasibility and unboundedness detection work on hand-constructed cases
- `pytest tests/test_pdhg.py` → all pass

---

### Ticket 1C-02: GPU PDHG (conditional — run GPU Go/No-Go gate first)

**File:** `solver/lp/pdhg_gpu.py` (only create if GPU Go/No-Go = GO)

**GPU Go/No-Go procedure (must execute before writing this file):**
1. Benchmark CPU PDHG (1C-01) vs CPU revised simplex (1B-01) on 5 largest Netlib instances
2. If GPU hardware available: port PDHG to CuPy (replace `np.` with `cp.` for array ops; matrix-vector becomes `cp.sparse.csr_matrix.dot`)
3. Benchmark GPU PDHG vs CPU simplex on same instances
4. **GO** if ≥2x wall-clock speedup on majority → feature as headline result with exact numbers
5. **NO-GO** if <2x or no GPU → keep CPU PDHG, document decision in `report/benchmark_report.md`:
   > "We implemented PDHG (Applegate et al. 2021) as a CPU-first first-order LP method. At Netlib problem sizes, our CPU revised simplex outperforms PDHG, consistent with the literature's finding that FOM/GPU advantage is size-dependent (cuPDLP.jl paper, 2023). PDHG is included as an architectural component and GPU-acceleration pathway for industrial-scale instances (>100k variables)."

**Verification gate:** GPU Go/No-Go decision documented in `benchmarks/results/gpu_decision.md` with timing numbers; if GO, `tests/test_pdhg_gpu.py` passes same T-05 checks.

---

## PHASE 2A — Presolve

**Goal:** Reduce problem size before solving. The #1 source of bugs is postsolve; implement both together.

---

### Ticket 2A-01: Presolve Reductions

**File:** `solver/presolve/presolve.py`

**Reductions to implement IN THIS ORDER** (cheapest/highest-value first):
1. **Empty row removal:** rows with no non-zero coefficients → check if `b >= 0` (feasible and can remove) or infeasible
2. **Empty column removal:** variables that appear in no constraint → fix to bound if objective pushes that way; remove
3. **Singleton row substitution:** row has exactly 1 non-zero → substitute variable value, propagate to objective and other constraints, remove row and variable
4. **Bound tightening from constraints:** for each row `aᵀx ≤ b`, tighten variable bounds using implied bounds from other variables' known ranges; iterate until fixed point
5. **Dominated column removal:** if a column's objective coefficient is non-improving in all constraints → fix to its bound
6. **Coefficient strengthening for binaries:** for binary `xᵢ` in `aᵀx ≤ b`, if `aᵢ > b - Σⱼ≠ᵢ aⱼ`, shrink `aᵢ` and adjust RHS (Achterberg et al. 2020 formula)
7. **Dual fixing:** if a constraint is always non-binding at any optimal (dual variable = 0 in a related relaxation), fix the corresponding variable

**Implementation rule:** Each reduction is a `(Problem, ReductionLog) -> (Problem, ReductionLog)` pure function. `ReductionLog` is a list of `(reduction_type, affected_var/row, original_value)` tuples needed for postsolve.

**Verification gate (T-06, partial):** Unit tests for each reduction type on tiny hand-crafted examples; each reduction preserves feasibility and objective on the toy.

---

### Ticket 2A-02: Postsolve

**File:** `solver/presolve/postsolve.py`

**Implementation steps:**
1. Implement `postsolve(presolved_solution, reduction_log) -> original_solution` that replays the `ReductionLog` in REVERSE ORDER
2. For each reduction type, implement the inverse mapping:
   - Substituted variable → recover from the substitution formula
   - Fixed variable → set to the fixed value
   - Removed row → reconstruct dual variable if needed (for dual warm-starts)
3. Return a solution vector in the original variable space

**Critical test — run after EVERY postsolve implementation change:**
```python
for instance in test_instances:
    prob = read_mps(instance)
    presolved_prob, log = presolve(prob)
    presolved_sol = solve(presolved_prob)
    original_sol = postsolve(presolved_sol, log)
    assert abs(eval_obj(prob, original_sol) - eval_obj(presolved_prob, presolved_sol)) < 1e-6
    assert is_feasible(prob, original_sol)
```

**Verification gate (T-06, full):**
- Zero objective-value mismatches on all test instances (original vs postsolved)
- Reduction counts logged: non-negative, reduced problem never larger than original
- Presolve ON vs OFF node-count comparison table (Phase 2B must be complete to produce this; create placeholder and fill in after 2B)
- `pytest tests/test_presolve.py` → all pass

---

## PHASE 2B — Basic MILP (Branch-and-Bound)

**Goal:** Working MILP solver. LP relaxations at each node. No cuts or advanced heuristics yet.

---

### Ticket 2B-01: B&B Tree + Node Data Structure

**File:** `solver/milp/branch_and_bound.py`

**Data structures:**
```python
@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    depth: int
    lb_changes: dict   # {var_idx: (new_lb, new_ub)} — bounds imposed at this node
    lp_basis: Optional[Basis]  # warm-start basis from parent LP
    lp_obj: float       # LP relaxation objective (lower bound for minimization)
    is_pruned: bool = False

class BnBTree:
    nodes: List[Node]
    open_nodes: List[int]  # node IDs available to explore
    best_incumbent: Optional[np.ndarray]
    best_obj: float = np.inf  # for minimization
```

**Node selection:** start with DFS (always take deepest unexplored node); implement best-first (priority queue by `lp_obj`) as a flag `node_selection='best_first'`; add plunging (dive until LP is infeasible or all integer, then switch to best-first) as `node_selection='hybrid'`.

**Verification gate:** tree data structure unit tests pass; `pytest tests/test_milp.py::test_tree_structure` passes.

---

### Ticket 2B-02: LP Bounding + Branch

**File:** `solver/milp/branch_and_bound.py` (add `solve_milp` function)

**Algorithm:**
```
solve_milp(problem):
  root_node = Node(id=0, lb_changes={})
  open_nodes = [root_node]
  best_obj = +inf; best_sol = None

  while open_nodes:
    node = select_node(open_nodes)                    # DFS / best-first
    lp = apply_bounds(problem, node.lb_changes)       # add var bounds from this node
    lp_sol = solve_lp(lp, warm_start=node.lp_basis)  # revised simplex (or dual simplex warm-start in 2D)

    if lp_sol.status == 'infeasible':
      prune(node); continue                           # infeasible subtree
    if lp_sol.obj >= best_obj - OPTIMALITY_TOL:
      prune(node); continue                           # bound pruning
    if is_integer_feasible(lp_sol.x, problem.integer_mask):
      update_incumbent(lp_sol.x, lp_sol.obj)
      prune(node); continue                           # new incumbent found
    
    branch_var = select_branch_variable(lp_sol, problem.integer_mask)  # most-fractional (Phase 2B)
    frac = lp_sol.x[branch_var]
    left_child = Node(lb_changes={**node.lb_changes, branch_var: (lb, floor(frac))})
    right_child = Node(lb_changes={**node.lb_changes, branch_var: (ceil(frac), ub)})
    open_nodes.extend([left_child, right_child])

  return best_sol, best_obj, optimality_gap
```

**Bound monotonicity assertion (development only, remove from production release):**
```python
assert node.lp_obj >= parent_node.lp_obj - 1e-6, "LP bound decreased after branching!"
```

**Verification gate (T-07):**
- 5-item knapsack toy: exact optimum, `is_feasible` passes
- All ≤20-binary-variable instances: 100% match with brute-force enumeration
- MIPLIB 10-instance subset: objective match where both prove optimal; gap reported otherwise
- Bound monotonicity assertion never fires during any test run
- `pytest tests/test_milp.py` → all pass

---

## PHASE 2C — Advanced Branching

**Goal:** Replace most-fractional branching with reliability pseudocost branching + on-the-fly learned ranker. Produce A/B/C comparison table.

---

### Ticket 2C-01: Reliability Pseudocost Branching

**File:** `solver/milp/branching.py`

**Implementation steps:**
1. Maintain per-variable pseudocost accumulators:
   ```python
   pscost_up[i] = sum(delta_obj_up[i]) / count_up[i]   # avg obj improvement per unit when branching up on var i
   pscost_down[i] = similar
   ```
2. Score variable `i`: `score(i) = max(pscost_up[i] * frac_up, EPSILON) * max(pscost_down[i] * frac_down, EPSILON)` (product score — Achterberg-Koch-Martin 2005 recommendation)
3. **Reliability check:** if `count_up[i] < RELIABILITY_THRESHOLD or count_down[i] < RELIABILITY_THRESHOLD` (from `config.py`), use strong branching for this variable to build up pseudocost history
4. **Strong branching:** for each unreliable candidate, solve left and right child LP relaxations (bounded to `STRONG_BRANCH_MAX_ITERS = 50` dual-simplex iterations); record actual delta_obj
5. **Update pseudocosts** after each branch

**Verification gate (T-08, partial):** Pseudocost accumulators correctly updated; strong branching produces correct delta_obj on toy instances.

---

### Ticket 2C-02: On-the-Fly Learned Branching Ranker

**File:** `solver/milp/branching.py` (add `LearnedRanker` class)

**Algorithm (Khalil et al. 2016, adapted):**
1. For the **first N nodes** of B&B (`LEARNED_RANKER_WARMUP = 200` nodes):
   - Compute features for each fractional variable candidate (see feature set below)
   - Record strong-branching score as the label
   - Collect `(features, strong_branch_score)` training pairs
2. After N nodes: fit a **ridge regression** model (`sklearn.linear_model.Ridge`) on the collected pairs
3. For all subsequent nodes: use the ridge regression to rank candidates instead of strong branching
4. The ranker is instance-specific (fit fresh per MILP instance)

**Feature set** (from Gasse et al. learn2branch `utilities.py`, simplified for from-scratch implementation):
- Fractionality: `frac = x[i] - floor(x[i])`
- Pseudocost up/down (from 2C-01)
- Pseudocost reliability count up/down
- Reduced cost of the variable
- Constraint degree (how many constraints the variable appears in)
- Objective coefficient normalized
- Current LP lower bound minus root LP lower bound (depth proxy)

**Verification gate (T-08, full):**
- A/B/C table: most-fractional vs pseudocost vs learned ranker on MIPLIB 10-instance subset
- All three rules produce correct solutions (is_feasible passes on all incumbents)
- Honest reporting: include instances where most-fractional wins
- `pytest tests/test_branching.py` → all pass

---

## PHASE 2D — Warm-Started Dual Simplex in B&B

**Goal:** Wire the dual simplex warm-start (from 1B-02) into the B&B node processing. Performance-only change; correctness must be re-confirmed.

---

### Ticket 2D-01: Warm-Start Integration

**File:** `solver/milp/branch_and_bound.py` (modify `solve_milp`)

**Implementation steps:**
1. When creating a child node, store the parent's optimal LP basis (`basis, basis_values`) in `Node.lp_basis`
2. When solving the child LP, call `dual_simplex_warm_start(basis, basis_values, bound_change)` instead of solving from scratch
3. **Fallback:** if dual simplex fails to find a feasible basis from warm-start within `WARMSTART_FAILOVER_ITERS = 200` iterations, fall back to cold-start primal simplex
4. **Re-run all T-07 tests after this change** — the correctness must be unchanged

**Verification gate (T-09):**
- T-07 correctness tests re-run: all still pass
- Wall-clock improvement table: warm-start vs cold-start on MIPLIB 10-instance subset
- Average improvement ≥ 2× speedup on LP solve time per node (expected from literature; document if not achieved)
- `pytest tests/test_milp.py` → still all pass after this change

---

## PHASE 3A — Cutting Planes

**Goal:** Add cut generation to the B&B tree (Branch-and-Cut). Generate MIR, cover, and clique cuts. Implement cut selection scoring.

---

### Ticket 3A-01: MIR Cut Generator

**File:** `solver/milp/cuts.py`

**Algorithm (MIR cuts):**
For a row `aᵀx ≤ b` with some integer variables:
1. Scale row to make RHS `b = f + n` where `n = floor(b)`, `f = b - n` (fractional part)
2. For each coefficient `aⱼ`: `āⱼ = aⱼ mod 1` if xⱼ is integer; `āⱼ = aⱼ / f` if continuous
3. MIR cut: `Σⱼ āⱼ xⱼ ≤ floor(b) + f`
4. Compute violation: how much the current LP solution violates this cut. Only add cut if violation > `CUT_VIOLATION_MIN` (from config.py)

**Cut selection scoring** (apply before adding cuts to LP):
- `score = violation × orthogonality × (1/density)`
- `orthogonality = min_j (1 - cos_angle(cut_j, cut_new))` vs already-added cuts
- `density = nnz(cut_coefficients) / n_vars`
- Add at most `MAX_CUTS_PER_ROUND` cuts per round, ranked by score

**Verification gate (T-10, partial):**
- On brute-force toy MILP (≤20 binaries): every generated cut is satisfied by all enumerated feasible integer points — assert this for EVERY cut
- LP relaxation bound strictly improves (or stays same) after cut round

---

### Ticket 3A-02: Cover Cuts

**File:** `solver/milp/cuts.py`

**Algorithm (Cover cuts for binary knapsack constraints):**
For a constraint `Σⱼ aⱼ xⱼ ≤ b` with `xⱼ ∈ {0,1}`:
1. A **cover** is a subset `C` such that `Σⱼ∈C aⱼ > b` (infeasible to set all to 1)
2. Cover inequality: `Σⱼ∈C xⱼ ≤ |C| - 1`
3. Find a cover by **greedy:** sort variables by `aⱼ` descending; take variables until sum exceeds `b`
4. **Lift** the cover inequality to strengthen it (use sequential lifting per Crowder-Johnson-Padberg 1983)
5. Only add if LP solution violates the cover cut

**Verification gate (T-10, partial):** Same validity check as MIR cuts.

---

### Ticket 3A-03: Clique Cuts

**File:** `solver/milp/cuts.py`

**Algorithm:**
1. Build conflict graph `G`: nodes = binary variables; edge `(i,j)` if `xᵢ + xⱼ ≤ 1` is implied (from a constraint `xᵢ + xⱼ ≤ 1` or from probing)
2. Find cliques: use a **greedy maximal clique finder** (not optimal — this is fine for a prototype):
   - Start from variable with highest LP relaxation value among binary vars
   - Iteratively add a neighbor that is connected to all current clique members and has high LP value
   - Stop when no such neighbor exists
3. For each clique `K`: cut `Σⱼ∈K xⱼ ≤ 1`
4. Add only cuts violated by the current LP solution

**Verification gate (T-10, full):**
- All three cut types: validity on toy instances, bound improvement on MIPLIB subset
- Node-count comparison: cuts ON vs OFF on MIPLIB 10-instance subset
- `pytest tests/test_cuts.py` → all pass

---

## PHASE 3B — Primal Heuristics

**Goal:** Find feasible incumbents faster. Implement in order: rounding → Feasibility Pump → RINS → Local Branching.

---

### Ticket 3B-01: Rounding / Diving Heuristic

**File:** `solver/milp/heuristics.py`

**Algorithm:**
1. Take the LP relaxation solution `x`
2. Round each integer variable to its nearest integer
3. Check `is_feasible(problem, rounded_x)` — if feasible, return as incumbent
4. **Diving variant:** fix one variable at a time (the most fractional one) to its rounded value, re-solve LP, repeat until integer or infeasible
5. Return `None` if no feasible solution found

**Verification gate (T-11, partial):** Feasibility checker called on every returned solution; zero violations.

---

### Ticket 3B-02: Feasibility Pump

**File:** `solver/milp/heuristics.py`

**Algorithm (Fischetti-Glover-Lodi 2005 + Achterberg-Berthold improvement):**
1. Solve LP relaxation → `x̄` (LP optimum)
2. Round: `x̃ = round(x̄[integer_vars])`
3. If `is_feasible(problem, x̃)` → return `x̃`
4. Solve a **repair LP**: `min Σᵢ |xᵢ - x̃ᵢ|` for integer vars `i` s.t. original constraints (use 1-norm linearization with auxiliary variables)
5. Let `x̄ = new LP solution`; round again → `x̃`
6. **Achterberg-Berthold improvement:** if stuck in cycle (same `x̃` repeated), add a random perturbation to the objective: `c_perturbed[i] = c[i] + ε * random()`
7. Repeat until feasible or `MAX_FP_ITERS = 100` iterations
8. **Guard:** always call `is_feasible(problem, x̃)` before returning

**Verification gate (T-11, partial):** FP finds a feasible solution on all 10 MIPLIB instances where a solution exists; every returned solution passes `is_feasible`.

---

### Ticket 3B-03: RINS

**File:** `solver/milp/heuristics.py`

**Algorithm (Danna-Rothberg-LePape 2005):**
1. Requires an incumbent `x*` and an LP relaxation solution `x̄`
2. Find variables where `x*[i] == round(x̄[i])` (they agree) → fix these to their value
3. Build a sub-MIP: original problem with agreed variables fixed
4. Call `solve_milp(sub_mip, node_limit=RINS_NODE_LIMIT)` — recursively uses the B&B solver with a budget
5. If sub-MIP finds a better solution → update incumbent
6. **Guard:** `is_feasible(problem, sub_sol)` before accepting

**Verification gate (T-11, partial):** RINS improves incumbent on ≥3 of the MIPLIB 10-instance subset (where initial incumbent was suboptimal).

---

### Ticket 3B-04: Local Branching

**File:** `solver/milp/heuristics.py`

**Algorithm (Fischetti-Lodi 2003):**
1. Requires an incumbent `x*`
2. Add a single Hamming-distance constraint: `Σᵢ |xᵢ - x*ᵢ| ≤ k` (linearized for binary vars: `Σᵢ (x*ᵢ - xᵢ) * x*ᵢ + (xᵢ - x*ᵢ) * (1 - x*ᵢ) ≤ k`)
3. Solve this smaller sub-MIP with `solve_milp(sub_mip, node_limit=LOCAL_BRANCH_NODE_LIMIT)`
4. If better solution found → update incumbent; expand `k` and repeat
5. **Guard:** `is_feasible(problem, sub_sol)` before accepting

**Verification gate (T-11, full):**
- All four heuristics: every returned incumbent passes `is_feasible`
- Time-to-first-solution improvement table: heuristics ON vs OFF on MIPLIB subset
- `pytest tests/test_heuristics.py` → all pass

---

## PHASE 4A — QP Engine

**Goal:** OSQP-style ADMM solver for convex QP. Can be developed in parallel with Phases 2-3.

---

### Ticket 4A-01: ADMM QP Solver

**File:** `solver/qp/admm.py`

**Problem:** `min ½xᵀPx + qᵀx s.t. l ≤ Ax ≤ u` (P must be PSD)

**ADMM algorithm (Stellato et al. 2020 — OSQP):**
1. Form KKT-like system: introduce auxiliary variable `z = Ax`, multiplier `y`
2. **ADMM iterations:**
   - x-update: solve `(P + ρ AᵀA) x = -q + Aᵀ(ρz - y)` using cached factorization of `(P + ρ AᵀA)`
   - z-update: `z^{new} = clip(Ax + y/ρ, l, u)` — simple projection
   - y-update: `y = y + ρ(Ax - z^{new})`
3. Cache factorization of `(P + ρ AᵀA)` before iterations (key performance trick — factor once, solve many times)
4. Adaptive `ρ`: if `||primal_residual|| >> ||dual_residual||`, increase ρ (and re-factor); if `<<`, decrease ρ
5. Convergence: `max(||Ax - z||, ||ρ Aᵀ(z - z_prev)||) < tol`
6. Infeasibility/unboundedness: track dual iterates diverging (per OSQP paper §3.4)

**Verification gate (T-12):**
- Toy QP: `min x²+y² s.t. x+y≥1` → `x=y=0.5, obj=0.5` (tol 1e-6)
- Cross-check vs `highspy` QP solver on 5 constructed QP instances (tol 1e-4)
- Correct infeasibility/unboundedness detection
- `pytest tests/test_qp.py` → all pass

---

## PHASE 4B — Indian Industrial Demo Models

**Goal:** Build, solve, and validate two domain-specific models that demonstrate industrial applicability.

---

### Ticket 4B-01: Crude Blending LP

**File:** `instances/demo/crude_blending.mps` + `benchmarks/demo_crude_blending.py`

**Model description:**
```
Variables: xᵢⱼ = volume of crude i blended into product stream j (continuous)
Objective: min Σᵢⱼ costᵢ * xᵢⱼ (minimize blending cost)
Constraints:
  - Mass balance: Σⱼ xᵢⱼ ≤ available_i (crude availability)
  - Product volume: Σᵢ xᵢⱼ ≥ demand_j (product demand)
  - Quality specs: Σᵢ quality_i * xᵢⱼ / Σᵢ xᵢⱼ ≤/≥ spec_j (linear approximation via normalization)
```
Use realistic but synthetic data: 5 crude types, 3 product streams, 2 quality properties (e.g., sulfur, API gravity).

**Sanity check (run before comparing to HiGHS):** relax all quality constraints → problem should reduce to "use cheapest crude" for each product; assert this holds programmatically.

**Verification gate (T-13, partial):**
- Our solver objective == HiGHS objective on same model (tol 1e-6)
- Sanity check passes
- Solution satisfies all constraints (via `is_feasible`)

---

### Ticket 4B-02: Refinery Scheduling MILP

**File:** `instances/demo/refinery_scheduling.mps` + `benchmarks/demo_refinery.py`

**Model description:**
```
Variables:
  yᵢₜ ∈ {0,1} = unit i runs in time period t
  xᵢₜ ≥ 0     = throughput of unit i in period t
Objective: min Σᵢₜ (startup_cost * yᵢₜ + operating_cost * xᵢₜ) - Σₜ revenue_t * output_t
Constraints:
  - Capacity: xᵢₜ ≤ capacity_i * yᵢₜ (big-M linkage: run = 0 → throughput = 0)
  - Minimum run time: yᵢₜ ≥ yᵢ,t-1 + startup_i,t (unit once started must run for min_run periods)
  - Mass balance: flow conservation across refinery units
  - Product demand: total output per period ≥ demand_t
```
Use 4 refinery units, 5 time periods → ~40 binary vars, ~100 constraints.

**Verification gate (T-13, full):**
- Our B&C solver finds a feasible solution within 120s (incumbent reported with optimality gap)
- If both solvers prove optimal: objectives match (tol 1e-6)
- Cover cuts (3A-02) and RINS (3B-03) both exercise on this model
- `pytest tests/test_demos.py` → all pass

---

## PHASE 5 — Benchmarking & Report

**Goal:** Generate all benchmark tables, plots, and the final report. Every number must trace to a CSV row.

---

### Ticket 5-01: LP Benchmark Harness

**File:** `benchmarks/run_lp_bench.py`

**Steps:**
1. Download all Netlib LP instances (use `urllib` to fetch from `https://www.netlib.org/lp/data/`)
2. For each instance, run:
   - Our primal revised simplex
   - Our IPM
   - Our PDHG
   - HiGHS (via `highspy`)
3. Log to `benchmarks/results/lp_results.csv`: `instance, solver, status, objective, wall_clock_s, iterations`
4. Time limit: 60s per solver per instance
5. **Assertion:** for every instance where both our solver and HiGHS report optimal: `|our_obj - highs_obj| / max(1, |highs_obj|) < 1e-6`

---

### Ticket 5-02: MILP Benchmark Harness

**File:** `benchmarks/run_milp_bench.py`

**Steps:**
1. Use MIPLIB 2017 benchmark set (download from `https://miplib.zib.de/`)
2. For each instance, run our B&C solver and HiGHS with 300s time limit
3. Log to `benchmarks/results/milp_results.csv`: `instance, solver, status, objective, gap, wall_clock_s, nodes`
4. **Assertion (when both prove optimal):** objective match to 1e-6

---

### Ticket 5-03: Comparison Tables

**File:** `benchmarks/run_comparison_tables.py`

**Produce three tables:**
1. **Presolve ON vs OFF:** run B&C on 10 MIPLIB instances with both settings; log node count, time, final gap
2. **Branching rules A/B/C:** most-fractional vs pseudocost vs learned ranker on same 10 instances; log node count, time
3. **Warm-start vs cold-start:** B&B with dual warm-start vs cold-start primal simplex; log LP-solve time per node, total time

---

### Ticket 5-04: Performance Profile Plots

**File:** `benchmarks/plot_performance.py`

**Standard OR performance profile (Dolan-Moré 2002):**
- For each solver, plot the fraction of instances solved within `τ × (best solver time for that instance)` vs `τ`
- A solver that wins on every instance has a curve that reaches 1.0 at `τ = 1.0`
- Generate separate plots for LP and MILP
- Save to `benchmarks/results/perf_profile_lp.png`, `benchmarks/results/perf_profile_milp.png`

---

### Ticket 5-05: Final Report

**File:** `report/benchmark_report.md`

**Sections:**
1. **Executive Summary:** sovereign motivation, what was built, top-line results
2. **Architecture:** module diagram (ASCII or mermaid), ADRs summary
3. **LP Results:** Netlib table + performance profile; simplex vs IPM vs PDHG comparison
4. **MILP Results:** MIPLIB table + performance profile; presolve/branching/warm-start tables
5. **Indian Industrial Demo:** crude blending and refinery scheduling results
6. **GPU Decision:** PDHG go/no-go outcome with reasoning and numbers
7. **Roadmap:** Phase 2 (MIQP/NLP), Phase 3 (GNN branching/Neural Diving), Phase 4 (GPU B&B, Benders, symmetry)
8. **Reproducibility:** all benchmark numbers trace to named CSV files; all instance downloads are scripted

**Gate:** every number in the report exists as a row in a CSV with a timestamp column. No estimated/hand-typed numbers.

---

## Cross-Cutting Concerns (apply throughout all phases)

### Agent Handoff Protocol
Every time a new agent (or a new context window) starts work on this project:
1. Read `.agents/CONTEXT.md` (domain glossary + ADRs)
2. Read `.agents/ACTIVE_SPEC.md` (full spec + test IDs)
3. Read this file (`master_implementation_plan.md`) to find the current phase and last completed ticket
4. Read `task.md` to see the current state of the TODO list
5. Run `pytest` to confirm the current passing state before making any changes

### Git Commit Protocol (per phase completion)
After each phase gate passes:
```bash
pytest                                     # confirm all pass
git add -A
git commit -m "phase <X>: <brief description>"
git push origin main
```

### Never-Skip Rules
- `is_feasible(problem, x)` called on EVERY heuristic output before accepting as incumbent
- `original_obj == postsolved_obj` checked after EVERY presolve change
- Every cut validity-checked against all enumerated feasible integer points on toy instances
- B&B bound monotonicity assertion active during development

---

## Current Status

| Phase | Status | Gate |
|---|---|---|
| 1A — Foundation | ✅ COMPLETE | `pytest tests/test_env.py` + T-01 + T-02 |
| 1B — LP Engine | ✅ COMPLETE | T-03 + T-04 (Revised Simplex + IPM) |
| 1C — PDHG | ✅ COMPLETE | T-05 (Chambolle-Pock First-Order) |
| 2A — Presolve | ✅ COMPLETE | T-06 (Presolve + Postsolve) |
| 2B — Basic B&B | ✅ COMPLETE | T-07 (Branch-and-Bound MILP) |
| 2C — Adv Branching | ✅ COMPLETE | T-08 (Pseudocost + Strong Branching) |
| 2D — Warm Start | ✅ COMPLETE | Simplex basis reuse in B&B |
| 3A — Cuts | ✅ COMPLETE | T-10 (MIR, Cover, Clique) |
| 3B — Heuristics | ✅ COMPLETE | T-11 (Rounding, Diving, FP, RINS) |
| 4A — QP ADMM | ✅ COMPLETE | T-12 (ADMM with KKT saddle system) |
| 4B — Demo Models | ✅ COMPLETE | T-13 (Crude Blending + Refinery Sched) |
| 5 — Benchmarks | ✅ COMPLETE | All CSVs + README benchmarks |
