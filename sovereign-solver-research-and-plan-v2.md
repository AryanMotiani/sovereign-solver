# Sovereign Optimization Solver — Research Pack v2 & Zero-Ambiguity Implementation Plan

This supersedes `sovereign-solver-research-and-plan.md` by (a) filling in every keyword cluster you listed with the actual research papers/repos behind it, including the Neural-Diving/Neural-Branching family sparked by arXiv:2012.13349, and (b) turning the implementation plan into a checklist-driven spec with, for every module: exact algorithm, exact test instances, exact pass/fail numbers, and exact "definition of done" — so an agent building this has no interpretive gaps and cannot silently hallucinate a fake "it works."

---

## PART A — Expanded Research Map (organized by your clusters)

### A.0 The arXiv:2012.13349 family (Neural Diving + Neural Branching) — and its lineage

| Paper | Link | What to extract |
|---|---|---|
| Nair, Bartunov, ... Zwols (Google DeepMind), "Solving Mixed Integer Programs Using Neural Networks" | arXiv:2012.13349 | Two learned components bolted onto SCIP: **Neural Diving** (NN produces partial integer assignments, remaining sub-MIP solved by SCIP) and **Neural Branching** (NN imitates a GPU-scaled variant of full strong branching). Reports 2x–10x (up to 10^5x on one dataset) primal-dual gap improvement over vanilla SCIP. This is the "ceiling" reference for what learned components can plausibly buy you — cite it in the pitch, but note it needs training data per instance family, which a hackathon prototype won't have time to generate at this scale. Use it to justify a *smaller*, from-scratch, feature-based (not deep-GNN) analogue instead (see A.2). |
| Precursor: Khalil, Le Bodic, Song, Nemhauser, Dilkina, "Learning to Branch in Mixed Integer Programming," AAAI 2016 | arXiv search "Learning to Branch in Mixed Integer Programming Khalil" (DOI 10.1609/AAAI.V30I1.10080) | Learns an SVM-based ranking function that imitates strong branching, **trained on-the-fly during a single B&B run** (no offline dataset needed) — this is the realistically implementable version for a hackathon: collect (features, strong-branching score) pairs from the first N nodes, fit a simple linear/ridge-regression ranker, switch to it for the rest of the tree. |
| Successor (GNN, sets SOTA vs SCIP defaults on large instances): Gasse, Chételat, Ferroni, Charlin, Lodi, "Exact Combinatorial Optimization with Graph Convolutional Neural Networks," NeurIPS 2019 | arXiv:1906.01629, code: **https://github.com/ds4dm/learn2branch** | Bipartite variable-constraint graph + GCN, imitation-learned from strong branching. Has a full open-source reproduction repo — read the repo's feature-extraction code (`utilities.py`) even if you don't train a GNN; the hand-engineered node/variable features it computes (objective coefficient, constraint degree, fractionality, pseudocost stats, etc.) are exactly the feature set worth using for the light-weight linear ranker above. |
| Standardized RL/ML environment for solver internals | Prieto, Gasse et al., "Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers," arXiv:2011.06069, **https://github.com/ds4dm/ecole** | Wraps SCIP's branching/node-selection/cut-selection decision points as Gym-style environments — good architecture reference for how to expose your own solver's decision points as a clean interface (`observe() -> features`, `action(choice)`), even though Ecole itself wraps SCIP and shouldn't be a dependency of your from-scratch solver. |
| Related learned-branching survey citation used by the above | Lodi & Zarpellon, "On learning and branching: a survey," TOP (2017) | One-paragraph survey to cite in the report for "prior art in ML-for-branching" without re-deriving everything above. |
| RL-based branching (2026, generalization-focused) | Mhamed et al., "Tree-Gate Proximal Policy Optimization (TGPPO)," AAAI 2026 workshop, ojs.aaai.org/index.php/AAAI/article/view/39619 | Shows RL (PPO) branching policies generalize better than imitation learning across heterogeneous instances — cite as "future work beyond imitation learning" in the roadmap. |
| GPU-scaled strong-branching-imitation for large instances | "Lookback for Learning to Branch," arXiv:2206.14987 | Good literature-review section (Section 2) that itself concisely summarizes the whole learning-to-branch lineage (Alvarez et al. 2017 extremely-randomized-trees variant, Gupta et al. 2020 GNN efficiency work) — a fast way to backfill citations without reading every individual paper. |

**Actionable takeaway:** implement the Khalil et al. (2016) style **on-the-fly learned ranking branching rule** as your ML differentiator (cheap, no external data needed, directly testable against reliability-branching node counts on the same instance) rather than attempting a trained GNN (Gasse et al.) or Neural Diving (Nair et al.) in a hackathon — those need pre-existing training datasets and GPUs for training that a 4-5 day prototype won't have time for, but should be named explicitly on the roadmap slide as "Phase 3: Gasse et al. GNN branching / Neural Diving, given a training corpus of Indian-industry MILP instances."

### A.1 Core MILP (architecture-level references)

| Search topic (yours) | Best resource found |
|---|---|
| "mixed integer linear programming solver" / "modern MILP solver architecture" | Achterberg, PhD Thesis, *Constraint Integer Programming*, TU Berlin 2007, DOI 10.14279/depositonce-1634 — the architecture reference (already in v1, restated as the anchor of this whole section). |
| "branch and cut MILP" / "branch and bound MILP" | Morrison, Jacobson, Sauppe, Sewell, "Branch-and-bound algorithms: A survey of recent advances in searching, branching, and pruning," *Discrete Optimization* 19 (2016) — a general B&B survey (not MILP-exclusive) good for the "search/tree" cluster below too. |
| "constraint integer programming" | Achterberg's own paper introducing the CIP paradigm: Achterberg, "Constraint Integer Programming," and the shorter companion Achterberg & Berthold, "SCIP - a framework to integrate Constraint and Mixed Integer Programming" (ZIB Report 04-19) — explains *why* SCIP models MIP as a special case of constraint programming, which is the conceptual reason its architecture is modular (each constraint type provides its own propagation/separation/branching callbacks). Useful if you want your solver's plugin architecture (presolve rules, cut generators, branching rules, primal heuristics as swappable modules) to mirror SCIP's — recommended, since it directly supports "modular architecture that can later be extended to MIQP/NLP/MINLP" from the problem statement. |
| Broad, very current (2024) survey tying all the sub-clusters together | "Branch and Bound in Mixed Integer Linear Programming Problems: A Survey of Techniques and Trends," arXiv:2111.06257 | **Single best "read this if you read nothing else in Part A" survey** — it explicitly covers node selection, branching (including ML-based), cutting planes (Gomory/MIR/cover/split/lift-and-project), and primal heuristics (feasibility pump, RINS, local branching, RENS, Crossover) all in one document with a shared notation. Use its bibliography as a checklist against the tables below to confirm nothing is missing. |

### A.2 Branching ⭐ (full cluster)

| Technique | Reference | Implementation note |
|---|---|---|
| Strong branching (the "ground truth" oracle everything else approximates) | Achterberg PhD thesis §5; also SCIP's `branch_fullstrong.c` (already listed v1) | For each fractional candidate, solve both child LP relaxations (bounded to a few dual-simplex iterations if warm-started) and score by bound improvement. Correct but expensive — O(#candidates) LP re-solves per node. |
| Pseudocost branching | Achterberg-Koch-Martin (2005), "Branching rules revisited," *OR Letters* 33(1):42-54 (already in v1) | Maintains running average of objective-per-unit-fractionality historically observed for each variable; O(1) per candidate after warm-up. |
| **Reliability (pseudocost) branching — recommended default** | Same Achterberg-Koch-Martin paper | Hybrid: use strong branching until a variable's pseudocost has been updated a "reliable" number of times (paper suggests ~4-8), then switch to pseudocost for that variable. This is what every serious solver (SCIP, Gurobi, CPLEX) defaults to — implement this, not naive most-fractional. |
| Approximate/GPU-scaled strong branching | Nair et al. arXiv:2012.13349 §3 (their "Neural Branching" imitates this variant) | If GPU available, batch-solve many candidate child LPs via the batched-PDHG technique in A.7 rather than sequential simplex re-solves. |
| Learning to branch (on-the-fly, no external data) | Khalil et al. 2016, AAAI (see A.0) | **Primary recommended ML differentiator** — see A.0 actionable takeaway. |
| Learning to branch (GNN, needs training corpus) | Gasse et al. 2019, code https://github.com/ds4dm/learn2branch | Stretch goal / roadmap item only. |
| Sparse learning for branching (2026, newest) | arXiv:2604.00094 (already in your original list) / Optimization Online 2026 | Confirmed still current — a sparsity-regularized learned scoring function; cite alongside Khalil et al. as "the 2016 idea has a 2026 sparse-learning successor," strengthens the "we track current research" narrative. |

### A.3 Presolve ⭐ (full cluster)

Already covered thoroughly in v1 §1.5 (Andersen & Andersen 1995; Achterberg-Bixby-Gu-Rothberg-Weninger 2020; Gamrath et al. 2015 "Progress in Presolving"; PaPILO paper+repo; 2026 variable-implication paper). Additions for your explicit sub-keywords:

| Sub-topic | Reference |
|---|---|
| "bound tightening" / "variable fixing" | Covered inside Achterberg et al. 2020 — specifically their "domain propagation" and "dual fixing" sections; also see Belotti, Cafieri, Lee, Liberti, "Feasibility-based bounds tightening via fixed points" (2010) for the pure bound-propagation algorithm in isolation (useful because it's a small, self-contained sub-algorithm, easy to implement first). |
| "probing" / "clique probing" | Already listed in v1 (Optimization Online 2026 "Clique Probing for Mixed Integer Programs," arXiv:2512.17551). Also foundational: Savelsbergh, "Preprocessing and Probing Techniques for Mixed Integer Programming," *ORSA J. Computing* 6(4), 1994 — the original probing paper (temporarily fix a binary variable to 0/1, propagate, see what else gets implied) — read this before the 2026 paper, it's the base case. |
| "constraint propagation MILP" | Standard CP literature term for the same bound-tightening ideas above; Achterberg's CIP thesis (A.1) frames presolve/propagation as a unified concept — useful for module design (one `propagate()` interface reused during presolve AND during B&B node processing, which is exactly what SCIP does). |
| "dominated columns" | Covered in Andersen & Andersen (1995) and Gamrath et al. (2015), already listed. |
| "coefficient strengthening" | Covered in Achterberg et al. 2020, §"Coefficient Strengthening" — a specific, cheaply implementable rule: for a binary variable in a `<=` constraint, if its coefficient exceeds what's needed to make the constraint redundant when the variable is 1, shrink the coefficient and adjust the RHS. Concrete formula is in the paper. |

### A.4 Cuts (full cluster)

| Cut family | Reference |
|---|---|
| Gomory (fractional & mixed-integer) cuts | Classic; modern treatment + computational results in Balas, Ceria, Cornuéjols, Natraj, "Gomory cuts revisited," *OR Letters* 19 (1996) (cited across the survey arXiv:2111.06257) |
| Mixed Integer Rounding (MIR) cuts — **recommended primary cut family to implement** | Nemhauser & Wolsey (1990) foundational; modern extension: Marchand & Wolsey on structured MIR. Survey summary with the exact derivation is in arXiv:2111.06257 §4 (already found above) — MIR is provably equivalent to Gomory mixed-integer cuts and split cuts, but has a cleaner derivation for from-scratch implementation (single-constraint aggregation + rounding, no simplex-tableau-row bookkeeping needed like Gomory). |
| Cover cuts (for knapsack-type / binary-with-capacity constraints — directly relevant to refinery capacity constraints) | Crowder, Johnson, Padberg, "Solving large-scale zero-one linear programming problems," *Operations Research* 31(5), 1983 (already surfaced in your presolve search results) |
| Clique cuts | Padberg, "On the facial structure of set packing polyhedra" (classic); implementable via a conflict graph (pairs of binary variables that can't both be 1) — build the graph, find maximal cliques (even a greedy heuristic clique-finder is fine for a prototype), each clique gives a valid `sum(x_i) <= 1` cut. |
| Learning to cut (ranking/selecting which cuts to add) | Already in your list: arXiv:2404.12638, "Learning to Cut" — use its cut-scoring formulation even for a hand-tuned (non-learned) scoring rule: score = (violation depth) x (orthogonality to existing cuts) x (density penalty for sparsity), the three criteria the paper itself cites from prior hand-designed cut-selection rules (this is literally what SCIP's default cut selector does, per Achterberg's thesis §5). |
| Conflict-graph-based cuts (has runnable reproducibility code) | arXiv:2410.15110 + **https://github.com/INFORMSJoC/2024.0999** (already in v1 — flagged again here because it's one of the only entries in this entire list with a directly runnable, paper-linked code artifact) |

### A.5 Search / Tree (node selection)

| Technique | Reference |
|---|---|
| Depth-first search (DFS) | Standard; use for prototype default — minimizes memory (one path of open nodes), good for finding feasible solutions fast, worse for proving optimality quickly on hard instances. |
| Best-first / best-bound search | Classic (Lawler & Wood 1966, "Branch-and-bound methods: a survey," *Operations Research* 14(4), — already surfaced above in your feasibility-pump search results). Minimizes total nodes explored to prove optimality but has worse memory behavior. |
| Hybrid (plunging / best-first with periodic dives) — **recommended for the prototype** | Achterberg PhD thesis §6 "Node Selection" — describes SCIP's actual default: best-first for the overall search, combined with depth-first "plunging" once a promising node is found, to get primal solutions quickly while keeping the best-bound guarantee for proving optimality. This single rule (a small `if` on remaining node quality) captures most of the practical benefit. |
| General parallel B&B taxonomy (context for the "advanced future work" cluster) | Gendron & Crainic, "Parallel Branch-And-Bound Algorithms: Survey and Synthesis," *Operations Research* 42(6), 1994 — the foundational survey, still cited by 2025 GPU B&B papers (see A.7); explains master-worker vs. distributed-tree parallelization patterns you'd choose between if extending to multi-core B&B. |

### A.6 Heuristics (primal heuristics — full cluster)

| Heuristic | Reference | Class |
|---|---|---|
| Feasibility Pump — **recommended first heuristic to implement** | Fischetti, Glover, Lodi, "The Feasibility Pump," *Math. Programming* 104(1), 2005 | Start heuristic — round LP relaxation, re-solve a distance-minimizing LP to the rounding, repeat; simple, no incumbent needed, directly implementable from the paper's 2-step iteration described in every survey above. |
| Improved Feasibility Pump | Achterberg & Berthold, "Improving the Feasibility Pump" (2005), opus4.kobv.de PDF already retrieved above | A small, concrete tweak (objective-perturbed rounding) that measurably improves solution quality on 89/121 tested instances per their own numbers — cheap upgrade once basic FP works. |
| Feasibility Pump 2.0 | Fischetti & Salvagnin, *Math. Prog. Computation* 1(2), 2009 | Further refinement — optional, only if time remains. |
| RINS (Relaxation Induced Neighborhood Search) | Danna, Rothberg, Le Pape, "Exploring relaxation induced neighborhoods to improve MIP solutions," *Math. Programming* 102, 2005 | Improve heuristic — needs an incumbent; fixes variables that agree between incumbent and LP relaxation, re-solves the smaller residual MIP (can literally call your own B&B recursively with a node/time budget). |
| Local Branching | Fischetti & Lodi, "Local branching," *Math. Programming* 98(1), 2003 | Improve heuristic — adds a single "Hamming-distance ≤ k" constraint around the incumbent and re-solves as a sub-MIP; simplest of the "solve a smaller MIP" heuristics to implement (only one linear constraint added). |
| Diving heuristics (generic family) | Covered in the arXiv:2111.06257 survey (already retrieved) | Round-and-fix-and-resolve-LP repeatedly along a single DFS path; simplest possible heuristic to implement first, before Feasibility Pump, as a sanity check that "some primal solution" comes out quickly. |
| Learning-based primal heuristics / Neural Diving | Nair et al., arXiv:2012.13349 (A.0); "Smart Feasibility Pump," arXiv:2102.09663 (RL variant of feasibility pump found above) | Roadmap-only items — name both in the pitch, implement neither in Phase 1 given data/training-time constraints. |

### A.7 Numerical Solver (LP engine internals — full cluster)

Already core content of v1 §1.1–1.2. Additional entries for your explicit sub-keywords:

| Sub-topic | Reference |
|---|---|
| "dual steepest edge" | Forrest & Goldfarb, "Steepest-edge simplex algorithms for linear programming," *Math. Programming* 57 (1992) — the standard reference for the pricing rule that outperforms naive Dantzig-rule pivoting; HiGHS's own docs (already linked v1) list this as one of its `simplex_dual_edge_weight_strategy` options — read the HiGHS option documentation itself (scipy's HiGHS wrapper docstring, already surfaced in your Huangfu/Hall search above) for a plain-English description of Dantzig vs. Devex vs. steepest-edge trade-offs before implementing. |
| "bound flipping ratio test" | Part of the Huangfu & Hall (2018) dual simplex paper already in v1 — the "long step" dual ratio test variant that lets multiple bound-flips happen per pivot; explicitly named and derived in that paper, don't need a separate reference. |
| "sparse LU factorization LP" | Suhl & Suhl, "A fast LU update for linear programming," *Annals of OR* 43 (1993) — standard reference for maintaining a factorized basis across simplex pivots without full refactorization (complements the Forrest-Tomlin/Bartels-Golub techniques already named in v1 §1.1). |
| "LP reoptimization" / "warm start simplex" | Directly what makes B&B-with-simplex practical: after branching (one bound change), resume the parent's optimal basis and dual-simplex re-optimize instead of solving from scratch — this is *the* reason dual simplex (not primal) is the standard LP engine inside B&B; explicitly discuss this design choice in your report even if a generic "resolve from scratch" fallback is used in the earliest prototype milestone. |
| Batched GPU LP solves for many related LPs at once (directly useful for GPU-accelerated strong branching, A.2) | **Blin, Gualandi, Maes, Lodi, Stellato (NVIDIA + academic co-authors), "Batched First-Order Methods for Parallel LP Solving in MIP," Jan 2026, arXiv:2601.21990** | **New, highly relevant find.** Extends PDHG to solve *batches* of related LPs in parallel on GPU using matrix-matrix instead of matrix-vector ops — explicitly targets exactly your two use cases: (1) strong-branching child-LP evaluation and (2) bound-tightening LPs inside presolve/propagation. This is the single best paper connecting your GPU-PDLP track (v1 §1.3) to your branching/presolve tracks — cite it directly as the technical basis for "GPU-accelerated strong branching" on the roadmap slide. Also cites Gurobi's own 2024 public position (Ed Rothberg webinar, see next row) that GPUs are now viable for LP via PDHG — good "even commercial vendors agree" citation for the pitch. |
| Industry skepticism/vindication timeline for GPU LP (good pitch-deck citation) | Gurobi support article, "docs on GPU support," support.gurobi.com/hc/en-us/articles/360012237852 | States plainly that Gurobi's own team was skeptical of GPUs for sparse LP through 2023, then in Dec 2024 demonstrated PDHG + barrier-on-GPU results on Grace Hopper hardware — directly corroborates that PDLP/PDHG (your novelty angle) is the *current, live* frontier, not a dead-end research curiosity. |

### A.8 Advanced / Future Work (full cluster — for the roadmap slide, not Phase 1 implementation)

| Topic | Reference |
|---|---|
| Symmetry handling / orbital fixing | Margot, "Exploiting orbits in symmetric ILP," *Math. Programming* 98(1-3), 2003 (original orbital fixing); Kaibel, Peinhardt, Pfetsch, "Orbitopal fixing," *Discrete Optimization* 8(4), 2011; Ostrowski, Linderoth, Rossi, Smriglio, "Orbital branching," *Math. Programming* 126(1), 2011; survey: Margot, "Symmetry in integer linear programming," in *50 Years of Integer Programming*, 2010. Also SCIP's own implementation docs: scipopt.org `symmetry_orbital.c` (already retrieved above) — good short, concrete description of the fix-by-orbit rule if you want a minimal implementable version. |
| Benders decomposition | Foundational: Benders (1962); modern logic-based extension: Hooker & Ottosson, "Logic-based Benders decomposition," *Math. Programming* 96(1), 2003; practical branch-and-cut implementation guide: Maher, "Implementing the branch-and-cut approach for a general purpose Benders' decomposition framework," *EJOR* 290(2), 2021; symmetry-aware 2026 extension: arXiv:2511.22251 (found above) — relevant to power-system/refinery scheduling decomposition (e.g. separate "which units run" master problem from "how much do they produce" subproblems). |
| Dantzig-Wolfe decomposition / column generation | Survey: Lübbecke & Desrosiers, "Selected Topics in Column Generation," *Operations Research* 53(6), 2005 (found above); broader reformulation survey covering both Benders and Dantzig-Wolfe together: Vanderbeck & Wolsey, "Reformulation and Decomposition of Integer Programs," in *50 Years of Integer Programming*, 2010 (found above) — this single survey is the best "if you only cite one decomposition reference" choice since it covers both techniques your keyword list asked about, side by side, with the branch-and-price extension to B&B. |
| Parallel branch and bound (CPU) | Gendron & Crainic (1994) survey (A.5, already listed). |
| GPU MILP solver / parallel B&B on GPU (all very recent, 2024-2026) | (1) "Design Considerations for GPU-based Mixed Integer Programming on Parallel Computing Platforms," ICPP 2021 workshop — sets the realistic expectations (GPUs help when the LP-relaxation matrix fits in one GPU's memory but the B&B tree is too big for a few nodes); (2) "From Sequential Nodes to GPU Batches: Parallel Branch and Bound for Optimal k-Sparse GLMs," arXiv:2605.22188 (2026) — concrete hybrid CPU-GPU B&B architecture (CPU owns tree logic, GPU owns batched dense numerical work) directly reusable as your architecture diagram; (3) "Performance and Portability in Multi-GPU Branch-and-Bound: Chapel vs CUDA/HIP," IPDPSW 2025 — multi-GPU scaling reference if the roadmap needs to claim multi-GPU, not just single-GPU. |

---

## PART B — Correctness & Verification Framework (so nothing is ever "hallucinated as working")

**Principle:** every module below has (1) a hand-computable toy instance with a known exact answer, (2) an automated cross-check against HiGHS on a bigger instance, and (3) an explicit numeric pass/fail threshold. A module is not "done" until all three are green. This section is the actual spec an agent should treat as tests-to-pass, not just a description.

### B.1 MPS Reader
- **Toy check:** hand-craft a 3-variable, 2-constraint `.mps` file by hand; assert the parsed `(c, A, b, bounds, sense)` matches your hand-written expected arrays exactly (`assert np.allclose(...)`).
- **Cross-check:** parse 5 real Netlib instances with your reader AND with `highspy`'s reader; assert identical `n_vars`, `n_constraints`, and objective coefficient vectors (allow float tolerance 1e-9).
- **Definition of done:** both checks pass on all 5 instances, zero exceptions on malformed/edge-case sections (RANGES, free rows, MARKER INTORG/INTEND for integer variables).

### B.2 LP Simplex (primal + dual)
- **Toy check:** solve the textbook 2-variable LP (e.g. `max 3x+5y s.t. x<=4, 2y<=12, 3x+2y<=18`) by hand (optimal at x=2,y=6, obj=36) — assert your solver returns exactly this to 1e-6.
- **Degenerate toy check:** construct a known degenerate LP (e.g. the classic Beale cycling example) and confirm your anti-cycling rule (Bland's rule) terminates in a bounded number of iterations rather than looping forever — assert `iterations < some_cap` and correct optimal value.
- **Cross-check:** solve all Netlib LP instances (~90 instances) with your solver and with `highspy`; assert `abs(your_obj - highs_obj) / max(1,abs(highs_obj)) < 1e-6` for every instance that both solvers report optimal on. Log (not fail) any instance where your solver times out — that's expected for some instances in Phase 1 and should be reported honestly in the benchmark table, not hidden.
- **Infeasible/unbounded checks:** hand-construct one infeasible LP and one unbounded LP; assert your solver correctly reports `INFEASIBLE` / `UNBOUNDED` status (not a wrong numeric answer) — this is a common and easy-to-miss correctness bug.
- **Definition of done:** 100% match with HiGHS objective values on Netlib instances your solver completes; correct status detection on the infeasible/unbounded toy cases; documented iteration counts and wall-clock vs. HiGHS in a results table (parity not required to pass, honest reporting is).

### B.3 Interior-Point Method
- Same Netlib cross-check as B.2, run independently through the IPM code path (not the simplex path) — this catches bugs specific to the IPM implementation that coincidentally don't show up in simplex.
- **Convergence check:** log the duality gap (`primal_obj - dual_obj`) per iteration and assert it's monotonically decreasing (or decreasing on average, allowing small IPM oscillation) — a non-decreasing gap indicates a sign error in the KKT system, the single most common IPM bug.
- **Definition of done:** matches HiGHS objective on the same Netlib subset as B.2 to 1e-6; gap trace behaves as expected; both simplex and IPM independently agree with each other on every shared toy/Netlib instance (this cross-check between your own two engines is itself a powerful correctness signal, since two independently-derived buggy implementations are unlikely to coincidentally agree).

### B.4 GPU PDLP/PDHG Engine
- **Toy check:** same 2-variable hand-solved LP as B.2; PDHG should converge to the same optimum within its tolerance (note: PDHG typically needs a looser tolerance, e.g. 1e-6 relative, and more iterations than simplex — document the iteration count).
- **Cross-check:** run on the MIPLIB-2017-derived LP relaxation set (the same 383-instance-style set the original PDLP paper benchmarks on, or a subset) — assert convergence to within 1e-6 relative accuracy of the HiGHS solution on instances where PDHG terminates within your iteration/time cap.
- **Regression check specific to FOMs:** verify infeasibility detection — construct one infeasible and one unbounded LP; the PDLP paper's iterates are supposed to diverge in a specific certifiable direction for infeasible/unbounded problems (per Applegate et al. §on infeasibility detection) — assert your implementation flags this rather than looping to the iteration cap silently.
- **Definition of done:** matches simplex/IPM objective values on the shared toy + a chosen Netlib/MIPLIB-LP subset; correctly flags infeasible/unbounded; report (not hide) the iteration count and wall-clock vs. simplex — first-order methods are expected to need more iterations, that's a known, explainable trade-off, not a bug.

### B.5 Presolve
- **Correctness invariant (the single most important test for presolve):** for every test instance, solve the ORIGINAL problem and the PRESOLVED problem independently; assert the two objective values agree to 1e-6 (after postsolve maps the presolved solution back to original-space variables). A presolve bug that silently changes the feasible region is the worst possible bug class here because it produces a plausible-looking but wrong "optimal" answer — treat any mismatch as a release-blocking failure, not a warning.
- **Reduction-count check:** on a MIPLIB subset, log (# variables removed, # constraints removed, # bound tightenings, # coefficient strengthenings) — assert these are non-negative and that the reduced problem is never larger than the original (a coding bug can accidentally add rows/columns).
- **Before/after benchmark:** the actual "prove presolve helps" experiment — run B&B with presolve ON vs OFF on the same MIPLIB subset with the same time limit; assert (and report, even if the effect is small in Phase 1) that ON does not perform worse in aggregate node count/time than OFF, per instance and on average.
- **Definition of done:** zero objective-value mismatches between original and presolved-then-postsolved solutions across the full test set; documented non-negative reduction counts; a results table showing presolve ON vs OFF.

### B.6 Branch-and-Bound (MILP)
- **Toy check:** hand-craft a tiny MILP (e.g. knapsack with 5 items, known optimal by enumeration) — assert your B&B returns the exact known-optimal objective and a genuinely integer-feasible solution vector (`assert np.all(x[integer_vars] == np.round(x[integer_vars]))`).
- **Exhaustive-enumeration cross-check on toy instances:** for instances small enough to brute-force (≤20 binary variables), enumerate all 2^n combinations, compute the true optimum, and assert your B&B matches it exactly — this is a zero-ambiguity ground truth check independent of any external solver.
- **Cross-check on MIPLIB subset:** run your B&B and HiGHS with the same wall-clock time limit (e.g. 300s); for instances **both** report as proven-optimal, assert objective match to 1e-6; for instances where only one proves optimality, report the best-found objective and optimality gap for both, honestly, in the results table (do not claim "solved" if only a feasible incumbent was found — always report the gap).
- **Bound correctness check (a subtle bug source):** at every node, assert the LP relaxation's objective is a valid bound (for minimization: LP relaxation objective ≤ true optimal of that subtree) — this can be checked automatically by also solving small subtrees to completion and comparing, or by asserting `parent_LP_bound <= child_LP_bound` (bounds must not decrease when branching adds constraints, for minimization) at every single node — instrument this as a running assertion during development, not just a final check.
- **Definition of done:** exact match on brute-forceable toy instances; correct optimality gap reporting (never silently wrong) on MIPLIB subset; the parent/child bound monotonicity assertion never fires during any test run.

### B.7 Reliability Pseudocost Branching / Learned Branching
- **A/B test, not a pass/fail correctness test (this module can't be "wrong" the way LP/MILP correctness can — it's a performance heuristic):** on the same MIPLIB subset with the same node/time limit, run B&B with (a) most-fractional branching, (b) reliability pseudocost branching, (c) your on-the-fly learned ranking rule; log node counts and wall-clock for each. **Definition of done is a reported comparison table, explicitly including cases where the fancier rule does NOT win** (this happens and is normal — reliability branching is well known to sometimes lose to simpler rules on small/easy instances due to per-node overhead) — do not cherry-pick only favorable instances in the report.

### B.8 Cutting Planes (MIR / cover / clique)
- **Validity check (non-negotiable):** for every generated cut, assert it does not exclude any known-feasible integer point — the simplest concrete test: for toy instances where you've enumerated all feasible integer points (B.6's brute-force set), assert every generated cut is satisfied by every one of those points. A cut that ever fails this test is a bug, full stop, regardless of any speed benefit it appears to give.
- **Effectiveness check:** measure LP-relaxation bound improvement immediately after adding a round of cuts, on a MIPLIB subset — assert the bound moves in the improving direction (tighter, i.e. increases for minimization) or stays the same, never loosens.
- **Definition of done:** zero validity failures on the enumerated toy set; documented bound-improvement numbers (even if modest) on the MIPLIB subset; cuts integrated into the B.6 B&B loop with a before/after node-count comparison.

### B.9 Primal Heuristics (Feasibility Pump, RINS, Local Branching)
- **Feasibility check (non-negotiable):** every solution returned by a heuristic must independently satisfy `A @ x <= b` (within tolerance) AND integrality for integer variables — write one shared `is_feasible(problem, x)` checker used to validate every heuristic's output before it's ever accepted as an incumbent. This single function is the most important guardrail in the whole heuristics module.
- **Effectiveness check:** on a MIPLIB subset, measure "time to first feasible solution" and "objective of first feasible solution vs. final proven optimum (if known)" with heuristics ON vs OFF.
- **Definition of done:** zero feasibility-checker failures across all runs; documented time-to-first-solution improvement table.

### B.10 QP (ADMM/OSQP-style)
- **Toy check:** hand-solve a small QP (e.g. `min x^2+y^2 s.t. x+y>=1`) — known optimum at x=y=0.5, obj=0.5 — assert match to 1e-6.
- **Cross-check:** compare against `highspy`'s QP solver (HiGHS supports convex QP) on a handful of constructed QP instances (e.g. Netlib LPs with an added `P = I` quadratic term) — assert objective match to 1e-4 (ADMM is typically lower-accuracy than IPM by default; document the accuracy/iteration trade-off rather than treating a looser but honestly-reported tolerance as a failure).
- **Infeasibility/unboundedness detection check**, same pattern as B.2/B.4.
- **Definition of done:** matches on toy + cross-check set within documented tolerance; correct status detection.

### B.11 Domain Demo Models (crude blending LP, refinery scheduling MILP)
- **Independent sanity check:** solve the SAME hand-built blending/scheduling model with `highspy` as well as your own solver; assert objective match — this is your strongest, most audience-legible correctness demonstration, so it must be airtight (no float mismatches, no silently-different constraint sets between the two solves).
- **Model sanity check (before even comparing solvers):** verify by hand that a few extreme/simple cases behave correctly — e.g. if all blend quality constraints are relaxed to be non-binding, the model should reduce to a trivial "use the cheapest crude" answer; assert this reduces correctly as a spot-check that the formulation itself (not just the solver) is right.
- **Definition of done:** your-solver-vs-HiGHS match on the real demo model; documented sanity-check behavior for at least one degenerate/simplified case.

---

## PART C — Hyper-Detailed, Checklist-Driven Implementation Plan

Each item below is phrased as a concrete deliverable + the verification step from Part B that must pass before moving on. Do not proceed to the next numbered item until the current one's checklist is fully checked.

### C.0 Environment & Harness (before any algorithm code)
- [ ] Python env with numpy, scipy, highspy (baseline only), pytest, matplotlib.
- [ ] `pip install highspy` verified working: solve one Netlib instance via highspy directly, print objective, sanity-check against the Netlib README's documented optimal value for that instance.
- [ ] Directory structure: `solver/` (your code), `tests/` (pytest files mirroring Part B), `benchmarks/` (harness scripts + results CSVs), `instances/` (downloaded MPS files), `report/` (generated tables/plots).
- [ ] Download script for: 5-10 Netlib LP instances (small, for fast iteration), 10-20 MIPLIB benchmark-set instances (mixed sizes), full sets held back for final numbers.
- [ ] **Gate:** `pytest tests/test_env.py` passes (a trivial test that just checks highspy + numpy import and solve a 1-variable LP).

### C.1 MPS Reader — see B.1 for tests
- [ ] Implement `read_mps(path) -> Problem` dataclass (`c, A_ub, b_ub, A_eq, b_eq, lb, ub, integer_mask, sense`).
- [ ] Handle: ROWS (N/L/G/E), COLUMNS, RHS, RANGES, BOUNDS (all bound types: UP/LO/FX/FR/MI/PL/BV), MARKER INTORG/INTEND.
- [ ] **Gate:** all B.1 tests green.

### C.2 Dense/Tableau Simplex (throwaway correctness scaffold)
- [ ] Implement the simplest possible tableau simplex (dense, no sparsity, no performance concern) purely to validate LP logic on tiny toy problems before investing in the sparse revised-simplex machinery.
- [ ] **Gate:** B.2's toy checks (both the 2-variable LP and the Beale cycling example with Bland's rule) pass on this scaffold.

### C.3 Revised Simplex (production LP engine, primal + dual)
- [ ] Implement explicit basis matrix `B`, its inverse (dense first, then swap to sparse LU), reduced costs, ratio test, Bland's rule fallback for degeneracy.
- [ ] Implement dual simplex variant (needed for warm-starting after a bound change — required later for B&B, see C.9).
- [ ] Swap dense basis inverse for sparse LU factorization + Forrest-Tomlin-style update (v1 §1.1) once correctness is established with the dense version — do NOT attempt sparse LU before the dense version passes all tests; isolate the sparsity optimization as its own diffable change.
- [ ] **Gate:** full B.2 checklist green (toy, degenerate toy, Netlib cross-check, infeasible/unbounded detection).

### C.4 Interior-Point Method
- [ ] Implement Mehrotra predictor-corrector per v1 §1.2 references, using the normal-equations reduction.
- [ ] **Gate:** full B.3 checklist green, INCLUDING the cross-agreement check against your own C.3 simplex on shared instances.

### C.5 GPU PDLP/PDHG
- [ ] Implement restarted PDHG per Applegate et al. (diagonal preconditioning, adaptive step size, adaptive restart) — CPU/NumPy version first, then CuPy port if a GPU is available at the hackathon.
- [ ] **Gate:** full B.4 checklist green.

### C.6 Presolve
- [ ] Implement in this order (cheapest/highest-value first, per Achterberg et al. 2020's own taxonomy): (1) empty row/column removal, (2) singleton row substitution, (3) bound tightening from constraints, (4) dominated column removal, (5) coefficient strengthening for binaries, (6) dual fixing.
- [ ] Implement postsolve (mapping a presolved-space solution back to original-space) IN LOCKSTEP with each presolve rule — never add a reduction without its corresponding postsolve step; this pairing is the #1 source of presolve bugs per the literature (Achterberg et al. 2020 devote significant space to postsolve correctness for exactly this reason).
- [ ] **Gate:** full B.5 checklist green, especially the non-negotiable original-vs-presolved objective match.

### C.7 Basic Branch-and-Bound
- [ ] DFS, most-fractional branching, LP-relaxation bounding via C.3.
- [ ] **Gate:** B.6's toy + brute-force-enumeration checks green (this must work perfectly on tiny instances before scaling up).
- [ ] Then add best-first / hybrid plunging node selection (A.5) — **Gate:** re-run B.6 cross-checks, confirm no regression, and log node-count improvement vs. plain DFS.

### C.8 Reliability Pseudocost Branching + On-the-Fly Learned Ranking
- [ ] Implement pseudocost tracking, reliability threshold logic, strong-branching fallback for unreliable variables (A.2).
- [ ] Implement the Khalil et al.-style on-the-fly ranker: collect (feature vector, strong-branching score) pairs for the first N nodes, fit ridge regression, switch over.
- [ ] **Gate:** full B.7 A/B/C comparison table produced and included in the report, with honest reporting of any instances where the simpler rule wins.

### C.9 Warm-Started Dual Simplex Inside B&B
- [ ] Wire C.3's dual simplex warm-start into C.7/C.8's node processing (parent optimal basis + one bound change -> dual re-optimize, not solve-from-scratch).
- [ ] **Gate:** re-run B.6 cross-checks (objective correctness unaffected), and separately log wall-clock improvement vs. cold-start-every-node as its own before/after table (this is a pure performance change, so its "test" is a timing comparison, not a correctness one — but the correctness gate must still be re-confirmed since a warm-start bug is a classic way to introduce silent wrong answers).

### C.10 Cutting Planes
- [ ] Implement MIR cuts first (A.4 rationale: simpler derivation than Gomory, same theoretical strength).
- [ ] Then cover cuts (for binary knapsack-like constraints — directly reusable in the refinery capacity-constraint demo model).
- [ ] Then clique cuts (conflict graph + greedy maximal clique search).
- [ ] Implement a simple cut-selection scoring rule (violation x orthogonality x sparsity, per A.4's "Learning to Cut" citation) rather than adding every generated cut unconditionally (unconditional addition bloats the LP and can slow things down — this is a documented, known failure mode, not a hypothetical).
- [ ] **Gate:** full B.8 checklist green (validity on brute-force toy set is non-negotiable; effectiveness numbers reported honestly).

### C.11 Primal Heuristics
- [ ] Rounding/diving heuristic first (simplest, sanity-check that "a feasible solution comes out fast").
- [ ] Feasibility Pump, then the Achterberg-Berthold improved variant if time remains.
- [ ] RINS and Local Branching (both call your own B.6/C.7 B&B recursively on a smaller sub-MIP with a node/time budget — reuse infrastructure, don't reimplement solving).
- [ ] **Gate:** full B.9 checklist green (the shared `is_feasible()` guard is mandatory and must be exercised by every heuristic's test).

### C.12 QP Engine
- [ ] Implement OSQP-style ADMM per v1 §1.6 (can be developed in parallel with C.6-C.11 by a second team member/track).
- [ ] **Gate:** full B.10 checklist green.

### C.13 Domain Demo Models
- [ ] Build crude-blending LP using Pochet & Wolsey-style formulation guidance (v1 §Phase 5), solved via your own C.3/C.4/C.5 engine.
- [ ] Build a simplified refinery unit on/off + flow MILP, solved via your own C.7-C.11 B&B stack, and specifically exercise the cover-cut (C.10) and RINS (C.11) modules on it since capacity/on-off structure is exactly what those techniques target.
- [ ] **Gate:** full B.11 checklist green.

### C.14 Benchmark Report Generation
- [ ] Automated harness: for every instance in the held-out final Netlib/MIPLIB sets, run (your solver, HiGHS) with identical time limits, log (status, objective, gap, wall-clock, node count where applicable) to a CSV.
- [ ] Generate a performance-profile plot (standard OR-literature style, already referenced v1) and a summary table.
- [ ] Explicitly generate the presolve-ON-vs-OFF, branching-rule-A-vs-B-vs-C, and warm-start-vs-cold-start comparison tables from C.6/C.8/C.9's gates — these are pitch-deck assets, not just internal test logs, so format them for the final report/slide deck directly.
- [ ] **Gate:** report file exists, every number in it traces back to a CSV row produced by an actual solver run (no hand-typed/estimated numbers in the final deck — if a number appears in the pitch, it must exist in a benchmark CSV with a timestamp).

---

## PART C.5 — GPU Go/No-Go Decision Gate (run this before claiming any GPU result)

The literature is clear that GPU-PDHG's advantage is **not uniform** — it grows with instance size and sparsity pattern (cuPDLP.jl paper explicitly notes "strong correlation between the GPU speed-up and the size of the instances," and Gurobi's own 2023→2024 reversal on GPU viability happened specifically because of very large LPs on Grace Hopper). On small-to-medium MIPLIB/Netlib instances — which is most of what a hackathon benchmark set contains — a well-implemented CPU simplex can easily beat a first-order GPU method. Deciding "GPU: yes/no" by feel risks either (a) burning a day on a GPU port that doesn't move the benchmark, or (b) dropping a genuinely good result because it wasn't tested properly. Use this instead:

**Step 1 — Build once, CPU-only, regardless of the final answer.** Implement PDHG (C.5) in plain NumPy first. This is required either way: even if GPU is a "no," a correct CPU PDHG implementation is still your evidence that you implemented a second, structurally different LP algorithm family from scratch — that claim survives independently of hardware. Do not skip this step while waiting to "decide about GPU."

**Step 2 — Timeboxed GPU port (hard cap: half a day).** Port the same PDHG code to CuPy (drop-in array-API swap for the matrix-vector/matrix-matrix operations — this is intentionally the cheapest possible GPU port, not a hand-written CUDA kernel). If CuPy/GPU hardware isn't available in the room at all, stop here — the answer is automatically "no," not because the algorithm is bad but because there's no hardware to run it on; say this plainly in the report rather than presenting untested performance claims.

**Step 3 — Pick the right test instances.** Select 3-5 of the **largest, sparsest** instances in your benchmark set (from MIPLIB LP relaxations or the largest Netlib instances) — small instances are the wrong test and will make GPU look artificially bad regardless of implementation quality. If you have access to a genuinely large synthetic instance (e.g., a big blending/network-flow LP you generate yourself), include it — this is closer to where the literature actually shows the effect.

**Step 4 — Apply the threshold.** Run your CPU revised simplex (C.3) and your GPU PDHG (C.5-ported) on the same instances with the same convergence tolerance (e.g. 1e-6 relative). Compute wall-clock ratio = simplex_time / gpu_pdhg_time per instance.

| Outcome | Decision |
|---|---|
| GPU PDHG achieves **≥2x** wall-clock speedup over your own CPU simplex on the majority of the large/sparse test instances | **GO.** Feature it as a headline result. Report the exact instances, sizes, and ratios — not a vague "GPU is faster" claim. |
| Speedup is **inconsistent, marginal (<2x), or GPU is slower** | **NO-GO for the benchmark claim.** Keep the CPU PDHG implementation in the codebase and mention it in the architecture/roadmap slide ("we implemented and tested a GPU-ready first-order method; at hackathon problem sizes CPU simplex remains faster, consistent with the literature's own finding that FOM/GPU advantage is size-dependent — this is expected and documented, not a failure"), but do **not** put a GPU speedup number on the results slide. |
| No GPU hardware available at all | **NO-GO by default**, same messaging as above minus the benchmark attempt — state hardware availability plainly rather than presenting simulated/estimated numbers. |

**Why 2x and not 1.1x:** a marginal win is within the noise of implementation quality differences between your from-scratch simplex and your from-scratch PDHG (neither is as tuned as HiGHS), so a small ratio doesn't actually tell judges anything about the algorithm — it's not a credible claim either way. 2x is roughly the low end of what published GPU-PDHG results report over *tuned commercial* baselines (cuPDLP.jl vs Gurobi, PDLP vs SCS at 6.3x geomean) — clearing 2x over your own untuned CPU baseline is a much lower bar than the papers clear over Gurobi, so it's a fair, achievable, and honest threshold rather than a cherry-picked one.

**One failure mode to guard against either way:** don't let this decision block Part C's critical path. C.3 (CPU simplex) and C.6-C.11 (presolve/B&B/cuts/heuristics) are the credibility layer and must be finished regardless of the GPU outcome — schedule the GPU port as parallel/optional work for a team member with slack time, not as a blocking dependency for anything else in the plan.

---

## PART D — Updated Machine-Readable Task List

```yaml
project: sovereign-optimization-solver
verification_policy: >
  No module is marked complete without its Part-B test suite passing.
  Every benchmark number presented externally must trace to a row in
  benchmarks/*.csv produced by an actual run, never a hand-estimated figure.
modules:
  - id: mps_reader
    tests: [B.1]
  - id: lp_simplex_dense_scaffold
    tests: [B.2.toy, B.2.degenerate_toy]
  - id: lp_simplex_revised_sparse
    algorithm: "primal+dual revised simplex, Bland's rule, Forrest-Tomlin LU update"
    reference: ["Vanderbei LP textbook", "Huangfu & Hall 2018 arXiv/DOI 10.1007/s12532-017-0130-5", "Suhl & Suhl 1993 fast LU update"]
    tests: [B.2.full]
  - id: lp_interior_point
    algorithm: "Mehrotra predictor-corrector, normal equations"
    reference: ["Mehrotra 1992", "Lustig-Marsten-Shanno 1992 DOI 10.1137/0802022"]
    tests: [B.3]
    cross_validates_against: [lp_simplex_revised_sparse]
  - id: lp_pdhg_gpu
    algorithm: "restarted PDHG, diagonal preconditioning, adaptive restart"
    reference: ["Applegate et al. arXiv:2106.04756", "cuPDLP.jl arXiv:2311.12180", "Batched FOM arXiv:2601.21990 for future batched-LP use in branching/presolve"]
    tests: [B.4]
    novelty_flag: true
    build_cpu_version: mandatory  # implement regardless of GPU decision
    gpu_port: conditional  # see Part C.5 Go/No-Go gate
    gpu_claim_threshold: ">=2x wall-clock speedup vs own CPU simplex, majority of large/sparse test instances"
    gpu_no_go_fallback: "keep CPU implementation, cite as architecture/roadmap item, no benchmark speedup number in final report"
  - id: presolve
    algorithm: "empty row/col removal, singleton substitution, bound tightening, dominated column removal, coefficient strengthening, dual fixing"
    reference: ["Andersen & Andersen 1995", "Achterberg-Bixby-Gu-Rothberg-Weninger 2020 DOI 10.1287/ijoc.2018.0857", "PaPILO arxiv 2206.10709 (architecture ref)"]
    tests: [B.5]
    non_negotiable_invariant: "original_obj == presolved_then_postsolved_obj (1e-6 tol)"
  - id: milp_branch_and_bound_basic
    algorithm: "DFS then best-first/plunging hybrid, most-fractional branching baseline"
    reference: ["Achterberg PhD thesis 2007 DOI 10.14279/depositonce-1634", "arXiv:2111.06257 B&B survey", "Gendron & Crainic 1994 parallel B&B survey (node-selection background)"]
    tests: [B.6]
  - id: milp_branching_advanced
    algorithm: "reliability pseudocost branching; on-the-fly learned ranking (ridge regression on strong-branching-imitation features)"
    reference: ["Achterberg-Koch-Martin 2005", "Khalil et al. 2016 AAAI (on-the-fly, no external data)", "Gasse et al. 2019 arXiv:1906.01629 (feature set reference, code github.com/ds4dm/learn2branch)", "arXiv:2604.00094 sparse learning 2026 (roadmap citation)"]
    tests: [B.7]
    roadmap_only: ["Nair et al. Neural Diving/Branching arXiv:2012.13349", "Gasse et al. full GNN training", "TGPPO RL branching (AAAI 2026 workshop)"]
  - id: cutting_planes
    algorithm: "MIR cuts primary; cover cuts; clique cuts via conflict graph; violation/orthogonality/sparsity cut selection scoring"
    reference: ["Nemhauser & Wolsey MIR", "Crowder-Johnson-Padberg 1983 cover cuts", "arXiv:2404.12638 Learning to Cut (scoring rationale)", "arXiv:2410.15110 + code github.com/INFORMSJoC/2024.0999"]
    tests: [B.8]
    non_negotiable_invariant: "every cut satisfied by every enumerated feasible integer point on toy instances"
  - id: primal_heuristics
    algorithm: "rounding/diving; Feasibility Pump (+ Achterberg-Berthold improvement); RINS; Local Branching"
    reference: ["Fischetti-Glover-Lodi 2005", "Achterberg-Berthold 2005 improved FP", "Danna-Rothberg-LePape 2005 RINS", "Fischetti-Lodi 2003 local branching"]
    tests: [B.9]
    non_negotiable_invariant: "every returned incumbent passes shared is_feasible() check"
    roadmap_only: ["Neural Diving arXiv:2012.13349", "Smart/RL Feasibility Pump arXiv:2102.09663"]
  - id: qp_admm
    algorithm: "OSQP-style operator splitting (ADMM)"
    reference: ["Stellato et al. 2020 arXiv:1711.08013", "osqp.org/docs/solver"]
    tests: [B.10]
  - id: domain_demo_models
    build_with: internal solver API only
    models: [crude_blending_LP, refinery_unit_scheduling_MILP]
    reference: ["Pochet & Wolsey Production Planning by MIP"]
    tests: [B.11]
    exercises_modules: [cutting_planes.cover_cuts, primal_heuristics.RINS]
  - id: symmetry_benders_columngen  # ROADMAP ONLY, not Phase 1
    status: roadmap_only
    reference: ["Margot 2003 orbital fixing", "Kaibel-Peinhardt-Pfetsch 2011 orbitopal fixing", "Ostrowski-Linderoth-Rossi-Smriglio 2011 orbital branching", "Hooker & Ottosson 2003 logic-based Benders", "Maher 2021 Benders branch-and-cut", "arXiv:2511.22251 symmetric Benders 2026", "Lübbecke & Desrosiers 2005 column generation survey", "Vanderbeck & Wolsey 2010 unified decomposition survey"]
  - id: gpu_parallel_bnb  # ROADMAP ONLY, not Phase 1
    status: roadmap_only
    reference: ["ICPP 2021 GPU MIP design considerations", "arXiv:2605.22188 hybrid CPU-GPU B&B 2026", "IPDPSW 2025 multi-GPU B&B Chapel/CUDA/HIP", "arXiv:2601.21990 batched FOM for strong branching/bound tightening on GPU"]
  - id: benchmark_harness
    inputs: [Netlib LP set, MIPLIB benchmark subset]
    compares_against: [highspy]
    outputs: [objective, gap, wall_clock, node_count, status] per instance
    required_comparison_tables: ["presolve ON vs OFF", "branching rule A vs B vs C", "warm-start vs cold-start B&B"]
explicit_constraint: >
  Do not import or call into CPLEX/Gurobi/Xpress/HiGHS/SCIP/CBC/GLPK inside the solver
  itself. These may only appear in the benchmark harness as comparison baselines.
no_hallucination_policy: >
  Every claim of "implemented" or "working" must reference a passing test ID from Part B.
  Every benchmark number quoted anywhere (report, deck, README) must trace to a specific
  row in a benchmarks/*.csv file with a timestamp and instance name. If a module is
  incomplete or a test is failing, state that explicitly rather than omitting the result.
```

---

*All new links added in this v2 pass were verified as of September 2026. arXiv IDs are permanent and resolvable at `https://arxiv.org/abs/<id>`. GitHub repos explicitly named (ds4dm/learn2branch, ds4dm/ecole, INFORMSJoC/2024.0999, scipopt/papilo) are the only ones in this document confirmed to have runnable, paper-linked code — all other "study the architecture" references point to production solver source trees (HiGHS, SCIP) for structural inspiration, not for direct reuse.*
