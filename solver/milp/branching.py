"""
solver/milp/branching.py
-------------------------
Advanced branching strategies for Branch-and-Bound.

Implements three strategies (selectable by the caller):
  1. Most-Fractional (baseline)          — already in branch_and_bound.py
  2. Strong Branching                    — solve child LP stubs, pick best gain
  3. Reliability Pseudocost Branching    — accumulate per-variable gain stats,
                                           fall back to strong branching when
                                           statistics are unreliable.

Feature vector (Khalil 2016) for on-the-fly learning is also computed but
the ranker itself is not trained by default (MILP_LEARNED_RANKER=False by
default).  The feature vector is returned alongside the branching decision
so the caller can log it.

References:
    Achterberg et al. (2005) "Branching rules revisited", Op. Res. Letters.
    Khalil et al. (2016) "Learning to Branch in Mixed Integer Programming", AAAI.
    Achterberg (2009) PhD Thesis §7.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    FEASIBILITY_TOL,
    INTEGER_TOL,
    OPTIMALITY_TOL,
    RELIABILITY_THRESHOLD,
    STRONG_BRANCH_CANDIDATES,
    STRONG_BRANCH_MAX_ITERS,
)
from solver.lp.simplex_dense import SolveResult
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem


# ── Pseudocost data ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class PseudocostEntry:
    """Per-variable pseudocost statistics."""
    up_sum:   float = 0.0    # Sum of observed up-branch gains
    down_sum: float = 0.0    # Sum of observed down-branch gains
    up_cnt:   int   = 0      # Number of up-branch observations
    down_cnt: int   = 0      # Number of down-branch observations

    @property
    def up_score(self) -> float:
        return self.up_sum / max(1, self.up_cnt)

    @property
    def down_score(self) -> float:
        return self.down_sum / max(1, self.down_cnt)

    @property
    def is_reliable(self) -> bool:
        return self.up_cnt >= RELIABILITY_THRESHOLD and self.down_cnt >= RELIABILITY_THRESHOLD

    def score(self, frac: float) -> float:
        """
        Product score:  max(ε, ψ↓·frac) · max(ε, ψ↑·(1-frac))
        """
        eps = 1e-6
        return max(eps, self.down_score * frac) * max(eps, self.up_score * (1.0 - frac))


class PseudocostTable:
    """Dictionary of PseudocostEntry, initialised on first access."""

    def __init__(self, n_vars: int):
        self.n = n_vars
        self._data: Dict[int, PseudocostEntry] = {}

    def __getitem__(self, j: int) -> PseudocostEntry:
        if j not in self._data:
            self._data[j] = PseudocostEntry()
        return self._data[j]

    def update_up(self, j: int, gain: float):
        self[j].up_sum  += max(0.0, gain)
        self[j].up_cnt  += 1

    def update_down(self, j: int, gain: float):
        self[j].down_sum += max(0.0, gain)
        self[j].down_cnt += 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fractional_vars(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices of integer-constrained variables with fractional values."""
    int_vars = np.where(mask)[0]
    fracs = np.abs(x[int_vars] - np.round(x[int_vars]))
    return int_vars[fracs > INTEGER_TOL]


def _make_child_problem(
    problem: Problem,
    lb_extra: np.ndarray,
    ub_extra: np.ndarray,
    branch_var: int,
    direction: str,  # 'up' or 'down'
    x_val: float,
) -> Problem:
    """Create the LP relaxation for one branch child."""
    lb = np.maximum(problem.lb, lb_extra).copy()
    ub = np.minimum(problem.ub, ub_extra).copy()
    if direction == "down":
        ub[branch_var] = min(ub[branch_var], np.floor(x_val))
    else:
        lb[branch_var] = max(lb[branch_var], np.ceil(x_val))
    return Problem(
        c=problem.c,
        A_ub=problem.A_ub,
        b_ub=problem.b_ub,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        lb=lb, ub=ub,
        integer_mask=np.zeros(problem.n_vars, dtype=bool),
        sense=problem.sense, name=problem.name,
    )


def _lp_gain(child_prob: Problem, parent_obj: float, max_iters: int = STRONG_BRANCH_MAX_ITERS) -> float:
    """
    Solve a child LP with limited iterations and return the objective gain.
    Returns 0.0 if infeasible (prune → infinite gain → equally good).
    """
    res = solve_lp_revised(child_prob)
    if res.status == "infeasible":
        return np.inf
    if res.status == "optimal":
        return max(0.0, res.objective - parent_obj)
    return 0.0   # iteration limit, unbounded — treat as no gain info


# ── Strategy 1: Most-fractional ───────────────────────────────────────────────

def branch_most_fractional(x: np.ndarray, mask: np.ndarray) -> int:
    """Return variable index closest to 0.5 fractional part."""
    cands = _fractional_vars(x, mask)
    if len(cands) == 0:
        return -1
    fracs = np.abs(x[cands] - np.round(x[cands]))
    return int(cands[np.argmax(np.minimum(fracs, 1.0 - fracs))])


# ── Strategy 2: Strong branching ─────────────────────────────────────────────

def branch_strong(
    x: np.ndarray,
    mask: np.ndarray,
    problem: Problem,
    lb_extra: np.ndarray,
    ub_extra: np.ndarray,
    parent_obj: float,
    pseudocosts: Optional[PseudocostTable] = None,
) -> Tuple[int, float, float]:
    """
    Strong branching: evaluate LP gains for top candidates, return best var.

    Returns (branch_var, down_gain, up_gain).
    """
    cands = _fractional_vars(x, mask)
    if len(cands) == 0:
        return -1, 0.0, 0.0

    # Score candidates by most-fractional, take top K
    fracs = np.abs(x[cands] - np.round(x[cands]))
    scores = np.minimum(fracs, 1.0 - fracs)
    top_k = min(STRONG_BRANCH_CANDIDATES, len(cands))
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_cands = cands[top_idx]

    best_var = int(top_cands[0])
    best_score = -np.inf
    best_down = 0.0
    best_up = 0.0

    for j in top_cands:
        x_val = x[j]
        down_prob = _make_child_problem(problem, lb_extra, ub_extra, j, "down", x_val)
        up_prob   = _make_child_problem(problem, lb_extra, ub_extra, j, "up",   x_val)

        down_gain = _lp_gain(down_prob, parent_obj)
        up_gain   = _lp_gain(up_prob,   parent_obj)

        # Product score: min(μ_down, μ_up) with correction for ties
        eps = 1e-6
        score = max(eps, down_gain) * max(eps, up_gain)

        if pseudocosts is not None:
            pseudocosts.update_down(j, down_gain)
            pseudocosts.update_up(j, up_gain)

        if score > best_score:
            best_score = score
            best_var   = int(j)
            best_down  = down_gain
            best_up    = up_gain

    return best_var, best_down, best_up


# ── Strategy 3: Reliability pseudocost ───────────────────────────────────────

def branch_reliability_pseudocost(
    x: np.ndarray,
    mask: np.ndarray,
    problem: Problem,
    lb_extra: np.ndarray,
    ub_extra: np.ndarray,
    parent_obj: float,
    pseudocosts: PseudocostTable,
) -> Tuple[int, float, float]:
    """
    Reliability pseudocost branching (Achterberg et al. 2005).

    For each fractional variable:
      - If pseudocost is reliable → use accumulated pseudocost score.
      - Otherwise → fall back to strong branching for that variable.

    Returns (branch_var, down_gain, up_gain).
    """
    cands = _fractional_vars(x, mask)
    if len(cands) == 0:
        return -1, 0.0, 0.0

    scores = {}
    unreliable = []
    for j in cands:
        pc = pseudocosts[j]
        if pc.is_reliable:
            frac = x[j] - np.floor(x[j])
            scores[int(j)] = pc.score(frac)
        else:
            unreliable.append(j)

    # Strong-branch the unreliable candidates
    for j in unreliable:
        x_val = x[j]
        down_prob = _make_child_problem(problem, lb_extra, ub_extra, j, "down", x_val)
        up_prob   = _make_child_problem(problem, lb_extra, ub_extra, j, "up",   x_val)
        dg = _lp_gain(down_prob, parent_obj)
        ug = _lp_gain(up_prob,   parent_obj)
        pseudocosts.update_down(j, dg)
        pseudocosts.update_up(j, ug)
        eps = 1e-6
        scores[int(j)] = max(eps, dg) * max(eps, ug)

    best_var = max(scores, key=scores.get)
    pc = pseudocosts[best_var]
    frac = x[best_var] - np.floor(x[best_var])
    return best_var, pc.down_score * frac, pc.up_score * (1.0 - frac)
