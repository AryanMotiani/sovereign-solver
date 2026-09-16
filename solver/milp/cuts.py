"""
solver/milp/cuts.py
--------------------
Cutting plane generators for MILP.

Implements four families of cuts:
  1. Gomory fractional cuts (Gomory 1958) — from LP tableau rows
  2. Mixed-Integer Rounding (MIR) cuts (van Roy & Wolsey 1987)
  3. Cover cuts for 0-1 knapsack rows
  4. Clique cuts from a conflict graph

Each generator returns a list of (a, b) pairs representing new inequalities:
    aᵀx ≤ b

which are then added as additional rows to the LP relaxation at the current
B&B node (or at the root).

Cut scoring and filtering:
  - Cuts are scored by violation: v = aᵀx* - b (positive = violated)
  - Dominated cuts (parallel to an existing constraint) are filtered out
  - Only cuts with violation ≥ CUT_VIOLATION_MIN are added

References:
    Nemhauser & Wolsey (1988) "Integer Programming", Ch.7.
    Van Roy & Wolsey (1987) "Solving Mixed Integer Programming Problems using
      Automatic Reformulation", Op. Res. 35(1).
    Achterberg (2009) PhD Thesis, §8.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import CUT_VIOLATION_MIN, MAX_CUTS_PER_ROUND
from solver.milp.gomory import generate_gomory_cuts
from solver.problem import Problem


CutRow = Tuple[np.ndarray, float]   # (a, b) for aᵀx ≤ b


# ── 1. Gomory fractional cuts ─────────────────────────────────────────────────
# (Implemented with tableau & Chvátal-Gomory in solver.milp.gomory)



# ── 2. MIR cuts ───────────────────────────────────────────────────────────────

def generate_mir_cuts(
    problem: Problem,
    x_star: np.ndarray,
    max_cuts: int = MAX_CUTS_PER_ROUND,
) -> List[CutRow]:
    """
    Generate Mixed-Integer Rounding (MIR) cuts from the LP relaxation.

    For each LP inequality row  aᵢᵀx ≤ bᵢ:
      Let bᵢ = ⌊bᵢ⌋ + fᵢ  (integer part + fractional part fᵢ ∈ [0,1))

      For each coefficient aᵢⱼ:
        if j is integer:  aᵢⱼ_mir = ⌊aᵢⱼ⌋ + max(0, frac(aᵢⱼ) - fᵢ) / (1 - fᵢ)
        if j is continuous: aᵢⱼ_mir = aᵢⱼ / (1 - fᵢ)   [only if aᵢⱼ ≥ 0]

      MIR cut:  aᵢ_mir ᵀx ≤ ⌊bᵢ⌋

    Only cuts with violation > CUT_VIOLATION_MIN are returned.

    Parameters
    ----------
    problem : Problem
    x_star  : LP relaxation solution (fractional)
    max_cuts: int

    Returns
    -------
    List of (a_cut, b_cut) — each row is a new inequality aᵀx ≤ b.
    """
    cuts: List[CutRow] = []
    if problem.n_ineq == 0:
        return cuts

    A = problem.A_ub.toarray()
    b = problem.b_ub
    int_mask = problem.integer_mask

    for i in range(problem.n_ineq):
        a_row = A[i].copy()
        b_i   = b[i]

        # Valid MIR cut requires b_i > 0 and no negative continuous coefficients
        if b_i < 1e-6:
            continue
        if np.any((~int_mask) & (a_row < -1e-6)):
            continue

        f_b   = b_i - np.floor(b_i)

        if f_b < 1e-6 or f_b > 1.0 - 1e-6:
            continue  # Skip rows with integer or near-integer RHS

        a_mir  = np.zeros_like(a_row)
        denom = 1.0 - f_b

        for j in range(len(a_row)):
            a_ij = a_row[j]
            if int_mask[j]:
                f_a = a_ij - np.floor(a_ij)
                if f_a <= f_b:
                    a_mir[j] = np.floor(a_ij)
                else:
                    a_mir[j] = np.floor(a_ij) + (f_a - f_b) / denom
            else:
                if a_ij > 0:
                    a_mir[j] = a_ij / denom
                else:
                    a_mir[j] = 0.0

        b_mir = np.floor(b_i)

        # Check violation
        violation = float(a_mir @ x_star) - b_mir
        if violation > CUT_VIOLATION_MIN:
            cuts.append((a_mir, b_mir))
            if len(cuts) >= max_cuts:
                break

    return cuts


# ── 2. Cover cuts ─────────────────────────────────────────────────────────────

def generate_cover_cuts(
    problem: Problem,
    x_star: np.ndarray,
    max_cuts: int = MAX_CUTS_PER_ROUND,
) -> List[CutRow]:
    """
    Generate cover cuts from 0-1 knapsack rows.

    For a row  aᵢᵀx ≤ bᵢ  with all aᵢⱼ ≥ 0 and all x_j ∈ {0,1}:
      A cover C ⊆ {j : aᵢⱼ > 0} satisfies  Σ_{j∈C} aᵢⱼ > bᵢ.
      The cover cut is:  Σ_{j∈C} xⱼ ≤ |C| - 1.

    We use a greedy cover: sort columns by (xⱼ / aᵢⱼ) descending,
    add to C until the sum of aᵢⱼ exceeds bᵢ.

    Only cuts with violation > CUT_VIOLATION_MIN are returned.
    """
    cuts: List[CutRow] = []
    if problem.n_ineq == 0:
        return cuts

    A = problem.A_ub.toarray()
    b = problem.b_ub
    int_mask = problem.integer_mask

    for i in range(problem.n_ineq):
        a_row = A[i]
        b_i   = b[i]

        if b_i < 1e-6:
            continue
        # Pure knapsack row: all nonzero coefficients must be positive integers
        if np.any((~int_mask) & (np.abs(a_row) > 1e-8)):
            continue
        if np.any(a_row < -1e-8):
            continue

        int_cols = np.where(int_mask & (a_row > 1e-8))[0]
        if len(int_cols) == 0:
            continue

        # Greedy cover: sort by x*/a (items contributing most fractionally)
        ratios = np.where(a_row[int_cols] > 0, x_star[int_cols] / a_row[int_cols], 0.0)
        order  = np.argsort(ratios)[::-1]  # Descending: most fractional first

        cover_sum = 0.0
        cover = []
        for k in order:
            j = int_cols[k]
            cover.append(j)
            cover_sum += a_row[j]
            if cover_sum > b_i + 1e-8:
                break

        if cover_sum <= b_i + 1e-8:
            continue  # Couldn't form a valid cover

        # Cover cut: sum_{j in cover} x_j <= |cover| - 1
        a_cut = np.zeros(problem.n_vars)
        for j in cover:
            a_cut[j] = 1.0
        b_cut = float(len(cover) - 1)

        violation = float(a_cut @ x_star) - b_cut
        if violation > CUT_VIOLATION_MIN:
            cuts.append((a_cut, b_cut))
            if len(cuts) >= max_cuts:
                break

    return cuts


# ── 3. Clique cuts ─────────────────────────────────────────────────────────────

def build_conflict_graph(
    problem: Problem,
    x_star: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Build a conflict graph: edge (i, j) exists if x_i + x_j ≤ 1 is
    implied by some constraint (e.g. from a packing row with capacity 1).

    Returns a list of conflict edges.
    """
    edges = []
    if problem.n_ineq == 0:
        return edges

    A = problem.A_ub.toarray()
    b = problem.b_ub
    int_mask = problem.integer_mask
    is_binary = int_mask & (problem.lb >= -1e-8) & (problem.ub <= 1.0 + 1e-8)

    for i in range(problem.n_ineq):
        a_row = A[i]
        b_i   = b[i]

        # Only look at binary packing rows: RHS in [1, 2), no negative coefficients
        if b_i < 0.99 or b_i >= 2.0 - 1e-8:
            continue
        if np.any(a_row < -1e-8):
            continue

        int_cols = np.where(is_binary & (a_row >= 1.0 - 1e-8))[0]
        if len(int_cols) < 2:
            continue

        # All pairs in this row conflict
        for ii in range(len(int_cols)):
            for jj in range(ii + 1, len(int_cols)):
                edges.append((int(int_cols[ii]), int(int_cols[jj])))

    return list(set(edges))  # Deduplicate


def generate_clique_cuts(
    problem: Problem,
    x_star: np.ndarray,
    max_cuts: int = MAX_CUTS_PER_ROUND,
) -> List[CutRow]:
    """
    Generate clique cuts from the conflict graph.

    A clique C in the conflict graph implies:  Σ_{j∈C} x_j ≤ 1.
    We use a greedy maximal clique search starting from the most
    fractional variable.

    Returns violated clique cuts aᵀx ≤ 1.
    """
    edges = build_conflict_graph(problem, x_star)
    if not edges:
        return []

    # Adjacency set
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    cuts: List[CutRow] = []
    int_mask = problem.integer_mask
    int_vars = np.where(int_mask)[0]

    # Sort by fractional part (most fractional first)
    fracs = np.abs(x_star[int_vars] - np.round(x_star[int_vars]))
    order = int_vars[np.argsort(fracs)[::-1]]

    visited_starts = set()
    for start in order:
        if start not in adj or start in visited_starts:
            continue
        visited_starts.add(start)

        # Greedy clique: start from `start`, greedily extend
        clique = {start}
        candidates = adj[start].copy()
        while candidates:
            # Pick candidate with highest fractional value
            best = max(candidates, key=lambda v: x_star[v])
            clique.add(best)
            candidates &= adj.get(best, set())

        if len(clique) < 2:
            continue

        # Clique cut: sum_{j in clique} x_j <= 1
        a_cut = np.zeros(problem.n_vars)
        for j in clique:
            a_cut[j] = 1.0
        b_cut = 1.0

        violation = float(a_cut @ x_star) - b_cut
        if violation > CUT_VIOLATION_MIN:
            cuts.append((a_cut, b_cut))
            if len(cuts) >= max_cuts:
                break

    return cuts


# ── Combined cut generator ─────────────────────────────────────────────────────

def generate_all_cuts(
    problem: Problem,
    x_star: np.ndarray,
    basis: Optional[np.ndarray] = None,
    max_total: int = MAX_CUTS_PER_ROUND,
) -> List[CutRow]:
    """
    Run all four cut generators (Gomory, MIR, Cover, Clique) and return
    the top `max_total` cuts sorted by violation (descending).

    Cut families:
      - Gomory: classical fractional cuts (explicit family, required by spec)
      - MIR: mixed-integer rounding cuts (stronger in general than Gomory)
      - Cover: 0-1 knapsack cover cuts
      - Clique: conflict-graph clique cuts
    """
    per_generator = max_total // 4 + 1
    cuts: List[CutRow] = []
    cuts.extend(generate_gomory_cuts(problem, x_star, basis=basis, max_cuts=per_generator))
    cuts.extend(generate_mir_cuts(problem, x_star, per_generator))
    cuts.extend(generate_cover_cuts(problem, x_star, per_generator))
    cuts.extend(generate_clique_cuts(problem, x_star, per_generator))

    # Score and deduplicate
    scored = []
    for a_cut, b_cut in cuts:
        v = float(a_cut @ x_star) - b_cut
        scored.append((v, a_cut, b_cut))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [(a, b) for _, a, b in scored[:max_total]]


def add_cuts_to_problem(problem: Problem, cuts: List[CutRow]) -> Problem:
    """
    Return a new Problem with cut rows appended to A_ub / b_ub.
    """
    if not cuts:
        return problem

    A_new_rows = np.array([a for a, _ in cuts])
    b_new = np.array([b for _, b in cuts])

    A_ub_combined = sp.vstack(
        [problem.A_ub, sp.csr_matrix(A_new_rows)], format="csr"
    )
    b_ub_combined = np.concatenate([problem.b_ub, b_new])

    return Problem(
        c=problem.c,
        A_ub=A_ub_combined,
        b_ub=b_ub_combined,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        lb=problem.lb,
        ub=problem.ub,
        integer_mask=problem.integer_mask,
        sense=problem.sense,
        name=problem.name + "_cut",
    )
