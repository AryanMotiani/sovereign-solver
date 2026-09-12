"""
solver/config.py
----------------
All named algorithm constants for the Sovereign Solver.
No magic numbers anywhere else in the codebase — every tunable parameter lives here.
"""

# ── Numerical tolerances ──────────────────────────────────────────────────────
PIVOT_TOL: float = 1e-7          # Minimum absolute value to accept as a pivot
FEASIBILITY_TOL: float = 1e-6   # Constraint / bound satisfaction tolerance
OPTIMALITY_TOL: float = 1e-6    # Reduced-cost / duality-gap convergence threshold
PDHG_TOL: float = 1e-6          # PDHG relative convergence tolerance
ADMM_TOL: float = 1e-4          # ADMM primal/dual residual tolerance (QP)
INTEGER_TOL: float = 1e-5       # |x - round(x)| threshold for integer feasibility

# ── Simplex ───────────────────────────────────────────────────────────────────
REFACTORIZE_EVERY: int = 50     # Re-factorize the basis matrix every N pivots
MAX_SIMPLEX_ITERS: int = 100_000
BLAND_RULE_THRESHOLD: int = 200  # Switch to Bland's rule after this many iterations
                                  # (anti-cycling safety net)

# ── Interior-Point Method ─────────────────────────────────────────────────────
MAX_IPM_ITERS: int = 200
IPM_BARRIER_COEFF: float = 0.1   # Mehrotra centering/correction blend factor

# ── PDHG ─────────────────────────────────────────────────────────────────────
MAX_PDHG_ITERS: int = 100_000
PDHG_RESTART_PERIOD: int = 50    # Check for adaptive restart every N iterations

# ── Branch-and-Bound ─────────────────────────────────────────────────────────
MAX_BB_NODES: int = 1_000_000
BB_TIME_LIMIT_S: float = 300.0   # Default 5-minute wall-clock limit per MILP solve
STRONG_BRANCH_CANDIDATES: int = 20   # Max fractional variables to evaluate with SB
STRONG_BRANCH_MAX_ITERS: int = 50    # Max dual-simplex iters per strong-branch LP
RELIABILITY_THRESHOLD: int = 8       # Min pseudocost updates before declaring reliable
LEARNED_RANKER_WARMUP: int = 200     # Nodes before on-the-fly ranker activates

# ── Cutting Planes ────────────────────────────────────────────────────────────
MAX_CUT_ROUNDS: int = 20            # Max rounds of cut generation per node
MAX_CUTS_PER_ROUND: int = 50        # Max cuts added per round (after scoring/filtering)
CUT_VIOLATION_MIN: float = 1e-4     # Minimum cut violation to bother adding

# ── Primal Heuristics ─────────────────────────────────────────────────────────
MAX_FP_ITERS: int = 100             # Feasibility Pump max iterations
RINS_NODE_LIMIT: int = 500          # Sub-MIP node budget for RINS
LOCAL_BRANCH_K: int = 20            # Initial Hamming-distance bound for Local Branching
LOCAL_BRANCH_NODE_LIMIT: int = 500  # Sub-MIP node budget for Local Branching

# ── ADMM / QP ─────────────────────────────────────────────────────────────────
ADMM_MAX_ITERS: int = 10_000
ADMM_RHO_INIT: float = 1.0          # Initial ADMM penalty parameter
ADMM_RHO_ADAPT_RATIO: float = 10.0  # Re-factor if rho changes by this ratio

# ── Warm-start fallover ───────────────────────────────────────────────────────
WARMSTART_FAILOVER_ITERS: int = 200  # Fall back to cold-start if warm-start stalls
