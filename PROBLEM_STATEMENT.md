# Smart India Hackathon (SIH) Problem Statement

## Title
**Development of a Sovereign Mathematical Optimization Solver Core**

---

## 1. Background
Almost every optimization problem in India's refining, petrochemical, power, logistics, manufacturing, and planning sectors ultimately depends on a handful of foreign mathematical optimization solvers such as IBM ILOG CPLEX, Gurobi, and FICO Xpress. These engines sit behind refinery scheduling, production planning, supply chain optimization, blending, energy management, and many AI-driven decision-support systems. 

While they are extremely capable, they come with:
- High recurring license costs
- Restrictive licensing models
- Limited visibility into the underlying optimization algorithms

Indian developers can formulate optimization problems, but they cannot inspect, modify, or tailor the solver internals to suit strategic national requirements. Open-source alternatives such as COIN-OR CBC, HiGHS, GLPK, and SCIP exist and have made significant progress, but they still lag behind commercial solvers for several classes of large-scale mixed-integer optimization problems and have not been developed, validated, or optimized specifically for Indian industrial use cases. 

The real challenge is not building the modeling interface; it is developing a **numerically robust optimization engine** that consistently finds high-quality solutions for large, sparse, and highly constrained industrial problems within practical computation times.

---

## 2. Description & Core Requirements
The objective is to develop a **sovereign mathematical optimization solver core** rather than a complete modeling environment.

### Solver Capabilities & Roadmap:
- **Initial Focus**:
  - Linear Programming (**LP**)
  - Mixed-Integer Linear Programming (**MILP**)
  - Quadratic Programming (**QP**)
- **Modular Extensibility**:
  - Mixed-Integer Quadratic Programming (**MIQP**)
  - Nonlinear Programming (**NLP**)
  - Mixed-Integer Nonlinear Programming (**MINLP**)

### Core Algorithmic Components:
- **Continuous Optimization**:
  - Revised Simplex Method (Primal and Dual Simplex with LU factorization and dynamic updating, e.g., Forrest-Tomlin or Bartels-Golub)
  - Interior-Point Methods (Primal-Dual Mehrotra Predictor-Corrector)
- **Mixed-Integer Optimization**:
  - Branch-and-Bound / Branch-and-Cut
  - Cutting Planes (Gomory fractional cuts, mixed-integer rounding, clique cuts, cover cuts)
  - Presolve & Bound Tightening (dual/primal reductions, probing)
  - Primal Heuristics (Feasibility Pump, RINS, Local Branching, Diving heuristics)
  - Advanced Node Selection Strategies (Best-Bound, Pseudo-Cost branching, Strong Branching)
- **Computational Engineering & Acceleration**:
  - Sparse Matrix Techniques & Numerical Linear Algebra (CSR/CSC, AMD/COLAMD reordering, sparse Cholesky / sparse LU)
  - Multi-core parallelization & thread-level concurrency
  - GPU acceleration considered where it provides measurable benefits

### Architectural Constraint:
> **Critical Requirement**: It shall **NOT** be built upon any existing open-source solver library (e.g., HiGHS, CBC, GLPK, SCIP), but shall be **built from scratch from mathematical foundation**.

---

## 3. Scope & Industrial Application Domains
The scope is to solve optimization problems arising from:
- Refinery scheduling & crude blending
- Process optimization & production planning
- Logistics & supply chain management
- Power system economic dispatch & unit commitment
- Transportation & infrastructure planning

**Benchmark Scale**: The solver should consistently deliver optimal or near-optimal solutions for industrial-scale problems involving thousands to millions of variables and constraints, including:
- Highly degenerate models
- Ill-conditioned constraint matrices
- Difficult mixed-integer formulations where weaker implementations exhibit excessive computation times or fail to converge.

---

## 4. Expected Solution & Deliverables
- **Core Engine & Interfaces**:
  - Robust optimization engine core.
  - Basic Application Programming Interface (API) (C++, Rust, or Python bindings) or Command-Line Interface (CLI).
  - Standard file format parsing (MPS, LP, QPS).
  - *Note*: A polished graphical user interface (GUI) is not required.
- **Benchmarking & Validation**:
  - Standard benchmark problems from recognised optimization libraries such as **Netlib** (LP), **MIPLIB** (MILP), and **Mittelmann** benchmark sets.
  - Solution quality and computational performance compared against at least one established solver.
- **Demonstration of Numerical Robustness**:
  - Successfully solving challenging large-scale optimization problems involving degeneracy, weak LP relaxations, or ill-conditioned constraint matrices.
- **Strategic Value**:
  - A transparent, extensible, and sovereign foundation for future Indian optimization software across industrial, scientific, and strategic applications.
