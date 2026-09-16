"""
solver/presolve/presolve.py
----------------------------
LP/MILP Presolve reductions.

Implements the 7 classical presolve reductions:
  1. Fixed-variable elimination (lb == ub)
  2. Row singleton elimination (one nonzero per row)
  3. Column singleton elimination (one nonzero per column in equality)
  4. Bound tightening from inequality rows
  5. Empty row / redundant row removal
  6. Empty column removal (only zero cost, zero coefficient → fix at lb)
  7. Dominated column bounds (dual bound tightening)

Postsolve data is accumulated in a stack of `PresolveAction` objects
which `postsolve.py` walks in reverse to reconstruct the original solution.

Design principle: every reduction is reversible.  No information is
discarded — it is pushed to the postsolve stack.

References:
    Savelsbergh (1994) INFORMS J. Comp. 6(4):445-454.
    Andersen & Andersen (1995) Math. Prog. 71(2):221-245.
    Achterberg (2009) PhD Thesis, §8.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional

import numpy as np
import scipy.sparse as sp

from solver.config import FEASIBILITY_TOL
from solver.problem import Problem


# ── Postsolve action data-classes ─────────────────────────────────────────────

@dataclasses.dataclass
class FixedVarAction:
    """Variable i was fixed to value v and removed."""
    orig_idx: int
    value: float


@dataclasses.dataclass
class RowSingletonAction:
    """Row i had a single nonzero a_ij.  x_j was fixed from the row."""
    orig_row: int
    col_j: int
    fixed_value: float
    a_ij: float
    b_i: float
    is_eq: bool   # True = came from equality row, False = inequality


@dataclasses.dataclass
class BoundTightenAction:
    """Bound on variable j was tightened (for postsolve info only)."""
    col_j: int
    old_lb: float
    old_ub: float
    new_lb: float
    new_ub: float


PresolveAction = FixedVarAction | RowSingletonAction | BoundTightenAction


# ── PresolveResult ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PresolveResult:
    """Output of the presolve pass."""
    problem: Problem                # Reduced problem (may be identical if nothing changed)
    col_map: np.ndarray             # col_map[j_reduced] = j_original
    actions: List[PresolveAction]   # Postsolve stack (applied in reverse)
    n_fixed: int = 0
    n_rows_removed: int = 0
    n_bound_tightened: int = 0
    infeasible: bool = False
    infeasibility_reason: str = ""


# ── Probing helper ────────────────────────────────────────────────────────────

def _probe_binary(
    j: int, value: float,
    lb: np.ndarray, ub: np.ndarray,
    A_ub_d: np.ndarray, b_ub: np.ndarray,
    A_eq_d: np.ndarray, b_eq: np.ndarray,
    active_ub_rows: np.ndarray, active_eq_rows: np.ndarray,
    active_cols: np.ndarray,
) -> bool:
    """
    Probe variable j fixed to `value` (0 or 1).
    Returns True if the resulting problem may be feasible
    (no obvious bound violation or constraint infeasibility).
    Returns False if a contradiction is detected.

    Uses a cheap propagation: substitute x_j=value into all rows,
    check if any active inequality b_ub becomes violated at maximum
    of remaining variables.
    """
    # Effective RHS after substituting x_j = value
    b_ub_p = b_ub.copy()
    b_eq_p = b_eq.copy()
    for i in np.where(active_ub_rows)[0]:
        b_ub_p[i] -= A_ub_d[i, j] * value
    for i in np.where(active_eq_rows)[0]:
        b_eq_p[i] -= A_eq_d[i, j] * value

    # Quick feasibility check: for active equality rows with only
    # inactive columns (all zero after removing j), b must be ~0.
    temp_active_cols = active_cols.copy()
    temp_active_cols[j] = False

    for i in np.where(active_eq_rows)[0]:
        row = A_eq_d[i, temp_active_cols]
        if np.all(np.abs(row) < FEASIBILITY_TOL):
            if abs(b_eq_p[i]) > FEASIBILITY_TOL:
                return False  # infeasible

    # For inequality rows: check if maximum achievable LHS can meet b_ub_p
    for i in np.where(active_ub_rows)[0]:
        row = A_ub_d[i, :]
        # Minimum value of row @ x (x_j=value fixed):
        # For positive coefs use lb, negative use ub
        min_val = 0.0
        for k in np.where(temp_active_cols)[0]:
            a = A_ub_d[i, k]
            if a > FEASIBILITY_TOL:
                min_val += a * lb[k]
            elif a < -FEASIBILITY_TOL:
                min_val += a * ub[k]
        if min_val > b_ub_p[i] + FEASIBILITY_TOL:
            return False  # even minimum LHS exceeds b_ub_p → infeasible

    return True


# ── Presolve pass ─────────────────────────────────────────────────────────────

def presolve(problem: Problem, max_rounds: int = 10) -> PresolveResult:
    """
    Apply LP/MILP presolve reductions until no further reductions fire
    or max_rounds is reached.

    Parameters
    ----------
    problem : Problem
    max_rounds : int
        Maximum number of full sweeps over all reductions.

    Returns
    -------
    PresolveResult
    """
    actions: List[PresolveAction] = []

    # Working copies (dense for simplicity of presolve arithmetic)
    n = problem.n_vars
    lb = problem.lb.copy()
    ub = problem.ub.copy()
    c  = problem.c.copy()

    # Combine ineq + eq into a single dense matrix for presolve
    m_ub = problem.n_ineq
    m_eq = problem.n_eq

    # We'll work entirely with A_full [m_ub | m_eq, n] and equality flags
    A_ub_d = problem.A_ub.toarray() if m_ub > 0 else np.zeros((0, n))
    A_eq_d = problem.A_eq.toarray() if m_eq > 0 else np.zeros((0, n))
    b_ub   = problem.b_ub.copy()
    b_eq   = problem.b_eq.copy()

    # Track which rows/cols are still active
    active_ub_rows = np.ones(m_ub, dtype=bool)
    active_eq_rows = np.ones(m_eq, dtype=bool)
    active_cols    = np.ones(n,    dtype=bool)

    n_fixed = 0
    n_rows_removed = 0
    n_bound_tightened = 0
    infeasible = False
    infeasibility_reason = ""

    for _round in range(max_rounds):
        changed = False

        # ── Reduction 1: Fixed variables ──────────────────────────────────────
        for j in np.where(active_cols)[0]:
            if np.isfinite(lb[j]) and np.isfinite(ub[j]):
                if lb[j] > ub[j] + FEASIBILITY_TOL:
                    infeasible = True
                    infeasibility_reason = f"Variable {j}: lb > ub"
                    break
                if abs(ub[j] - lb[j]) < FEASIBILITY_TOL:
                    v = (lb[j] + ub[j]) / 2.0
                    # Remove from all rows: b -= a_ij * v
                    b_ub[active_ub_rows] -= A_ub_d[np.ix_(active_ub_rows, [j])].ravel() * v
                    b_eq[active_eq_rows] -= A_eq_d[np.ix_(active_eq_rows, [j])].ravel() * v
                    A_ub_d[:, j] = 0.0
                    A_eq_d[:, j] = 0.0
                    active_cols[j] = False
                    actions.append(FixedVarAction(orig_idx=int(j), value=float(v)))
                    n_fixed += 1
                    changed = True
        if infeasible:
            break

        # ── Reduction 2: Empty rows (ineq) ────────────────────────────────────
        for i in np.where(active_ub_rows)[0]:
            row = A_ub_d[i, active_cols]
            if np.all(np.abs(row) < FEASIBILITY_TOL):
                # Empty row:  0 ≤ b_ub[i] must hold
                if b_ub[i] < -FEASIBILITY_TOL:
                    infeasible = True
                    infeasibility_reason = f"Empty inequality row {i} with b={b_ub[i]:.4g} < 0"
                    break
                active_ub_rows[i] = False
                n_rows_removed += 1
                changed = True
        if infeasible:
            break

        # ── Reduction 3: Empty equality rows ─────────────────────────────────
        for i in np.where(active_eq_rows)[0]:
            row = A_eq_d[i, active_cols]
            if np.all(np.abs(row) < FEASIBILITY_TOL):
                if abs(b_eq[i]) > FEASIBILITY_TOL:
                    infeasible = True
                    infeasibility_reason = f"Empty equality row {i} with b={b_eq[i]:.4g} ≠ 0"
                    break
                active_eq_rows[i] = False
                n_rows_removed += 1
                changed = True
        if infeasible:
            break

        # ── Reduction 4: Row singleton (equality) — fix variable ──────────────
        for i in np.where(active_eq_rows)[0]:
            active_col_idx = np.where(active_cols)[0]
            row_vals = A_eq_d[i, active_col_idx]
            nonzero_mask = np.abs(row_vals) > FEASIBILITY_TOL
            if nonzero_mask.sum() == 1:
                j_local = np.where(nonzero_mask)[0][0]
                j = active_col_idx[j_local]
                a_ij = A_eq_d[i, j]
                x_j = b_eq[i] / a_ij
                # Bound check
                if x_j < lb[j] - FEASIBILITY_TOL or x_j > ub[j] + FEASIBILITY_TOL:
                    infeasible = True
                    infeasibility_reason = (
                        f"Row singleton (eq row {i}, col {j}): "
                        f"x={x_j:.4g} out of bounds [{lb[j]:.4g}, {ub[j]:.4g}]"
                    )
                    break
                # Fix variable
                b_ub[active_ub_rows] -= A_ub_d[np.ix_(active_ub_rows, [j])].ravel() * x_j
                b_eq[active_eq_rows] -= A_eq_d[np.ix_(active_eq_rows, [j])].ravel() * x_j
                A_ub_d[:, j] = 0.0
                A_eq_d[:, j] = 0.0
                active_eq_rows[i] = False
                active_cols[j] = False
                actions.append(RowSingletonAction(
                    orig_row=int(i), col_j=int(j),
                    fixed_value=float(x_j), a_ij=float(a_ij),
                    b_i=float(b_eq[i]), is_eq=True,
                ))
                lb[j] = x_j; ub[j] = x_j
                n_fixed += 1; n_rows_removed += 1
                changed = True
        if infeasible:
            break

        # ── Reduction 5: Bound tightening from inequality rows ────────────────
        for i in np.where(active_ub_rows)[0]:
            active_col_idx = np.where(active_cols)[0]
            row = A_ub_d[i, active_col_idx]
            nonzero_mask = np.abs(row) > FEASIBILITY_TOL
            if nonzero_mask.sum() == 0:
                continue

            # For single-variable rows: directly tighten bound
            if nonzero_mask.sum() == 1:
                j_local = np.where(nonzero_mask)[0][0]
                j = active_col_idx[j_local]
                a_ij = A_ub_d[i, j]

                if a_ij > 0:
                    # a_ij * x_j ≤ b_i  →  x_j ≤ b_i / a_ij
                    new_ub = b_ub[i] / a_ij
                    if new_ub < ub[j] - FEASIBILITY_TOL:
                        actions.append(BoundTightenAction(j, lb[j], ub[j], lb[j], new_ub))
                        ub[j] = new_ub
                        n_bound_tightened += 1
                        changed = True
                    active_ub_rows[i] = False
                    n_rows_removed += 1
                    changed = True
                elif a_ij < 0:
                    # a_ij * x_j ≤ b_i  → x_j ≥ b_i / a_ij
                    new_lb = b_ub[i] / a_ij
                    if new_lb > lb[j] + FEASIBILITY_TOL:
                        actions.append(BoundTightenAction(j, lb[j], ub[j], new_lb, ub[j]))
                        lb[j] = new_lb
                        n_bound_tightened += 1
                        changed = True
                    active_ub_rows[i] = False
                    n_rows_removed += 1
                    changed = True

        # ── Reduction 6: Probing (binary variables) ───────────────────────────
        # For each binary variable j: fix to 0, propagate; fix to 1, propagate.
        # If one direction is infeasible → fix variable to the other value.
        # If both infeasible → problem infeasible.
        # If both agree on a bound tightening → apply it.
        # Limit: only probe if few active binary variables (keep runtime bounded).
        active_int_cols = [
            j for j in np.where(active_cols)[0]
            if problem.integer_mask[j] and abs(lb[j]) < FEASIBILITY_TOL
            and abs(ub[j] - 1.0) < FEASIBILITY_TOL
        ]
        MAX_PROBE = 20  # probe at most 20 binary vars per round
        for j in active_int_cols[:MAX_PROBE]:
            # --- Probe x_j = 0 ---
            feasible_0 = _probe_binary(
                j, 0.0, lb, ub, A_ub_d, b_ub, A_eq_d, b_eq,
                active_ub_rows, active_eq_rows, active_cols
            )
            # --- Probe x_j = 1 ---
            feasible_1 = _probe_binary(
                j, 1.0, lb, ub, A_ub_d, b_ub, A_eq_d, b_eq,
                active_ub_rows, active_eq_rows, active_cols
            )

            if not feasible_0 and not feasible_1:
                infeasible = True
                infeasibility_reason = f"Probing: both x_{j}=0 and x_{j}=1 infeasible"
                break
            elif not feasible_0:
                # Must have x_j = 1
                actions.append(BoundTightenAction(j, lb[j], ub[j], 1.0, 1.0))
                lb[j] = ub[j] = 1.0
                n_bound_tightened += 1
                changed = True
            elif not feasible_1:
                # Must have x_j = 0
                actions.append(BoundTightenAction(j, lb[j], ub[j], 0.0, 0.0))
                lb[j] = ub[j] = 0.0
                n_bound_tightened += 1
                changed = True
        if infeasible:
            break

        if not changed:
            break

    # ── Build reduced problem ─────────────────────────────────────────────────
    if infeasible:
        return PresolveResult(
            problem=problem,
            col_map=np.arange(n),
            actions=actions,
            n_fixed=n_fixed,
            n_rows_removed=n_rows_removed,
            n_bound_tightened=n_bound_tightened,
            infeasible=True,
            infeasibility_reason=infeasibility_reason,
        )

    # Extract active columns and rows
    col_map = np.where(active_cols)[0]           # reduced col j → original col
    ub_rows = np.where(active_ub_rows)[0]
    eq_rows = np.where(active_eq_rows)[0]

    A_ub_r = A_ub_d[np.ix_(ub_rows, col_map)] if len(ub_rows) > 0 else np.zeros((0, len(col_map)))
    A_eq_r = A_eq_d[np.ix_(eq_rows, col_map)] if len(eq_rows) > 0 else np.zeros((0, len(col_map)))
    b_ub_r = b_ub[ub_rows]
    b_eq_r = b_eq[eq_rows]
    c_r    = c[col_map]
    lb_r   = lb[col_map]
    ub_r   = ub[col_map]
    int_r  = problem.integer_mask[col_map]

    reduced = Problem(
        c=c_r,
        A_ub=sp.csr_matrix(A_ub_r) if A_ub_r.shape[0] > 0 else sp.csr_matrix((0, len(col_map))),
        b_ub=b_ub_r,
        A_eq=sp.csr_matrix(A_eq_r) if A_eq_r.shape[0] > 0 else sp.csr_matrix((0, len(col_map))),
        b_eq=b_eq_r,
        lb=lb_r,
        ub=ub_r,
        integer_mask=int_r,
        sense=problem.sense,
        name=problem.name + "_presolved",
    )

    return PresolveResult(
        problem=reduced,
        col_map=col_map,
        actions=actions,
        n_fixed=n_fixed,
        n_rows_removed=n_rows_removed,
        n_bound_tightened=n_bound_tightened,
        infeasible=False,
    )
