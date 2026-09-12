"""
solver/presolve/postsolve.py
----------------------------
Postsolve: reconstruct the original solution from the reduced-problem solution
by replaying the PresolveAction stack in reverse order.

The postsolve stack is a list of PresolveAction objects produced by
presolve.presolve().  We walk it in reverse (last-in, first-out) and
apply each action's inverse to build up the full solution vector.
"""

from __future__ import annotations

import numpy as np

from solver.presolve.presolve import (
    BoundTightenAction,
    FixedVarAction,
    PresolveResult,
    RowSingletonAction,
)


def postsolve(result: PresolveResult, x_reduced: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original-space solution from the reduced solution.

    Parameters
    ----------
    result : PresolveResult
        Output of presolve.presolve(), containing the action stack and col_map.
    x_reduced : np.ndarray
        Optimal solution for result.problem (the reduced problem).

    Returns
    -------
    np.ndarray
        Solution in the original variable space.
    """
    # Determine original problem size from col_map and actions
    n_orig = _infer_n_orig(result)
    x_full = np.zeros(n_orig)

    # Place the reduced solution back into original positions
    for j_red, j_orig in enumerate(result.col_map):
        if j_red < len(x_reduced):
            x_full[j_orig] = x_reduced[j_red]

    # Walk action stack in reverse order
    for action in reversed(result.actions):
        if isinstance(action, FixedVarAction):
            x_full[action.orig_idx] = action.value

        elif isinstance(action, RowSingletonAction):
            # Variable was fixed from a row singleton: value is already stored
            x_full[action.col_j] = action.fixed_value

        elif isinstance(action, BoundTightenAction):
            # Bound tightening is informational only — clip the value to
            # the ORIGINAL bounds (undo the tightening for reporting)
            j = action.col_j
            x_full[j] = np.clip(x_full[j], action.old_lb, action.old_ub)

    return x_full


def _infer_n_orig(result: PresolveResult) -> int:
    """
    Infer the original number of variables from the col_map and actions.
    """
    n_candidates = [int(result.col_map.max()) + 1] if len(result.col_map) > 0 else [0]
    for action in result.actions:
        if isinstance(action, (FixedVarAction,)):
            n_candidates.append(action.orig_idx + 1)
        elif isinstance(action, (RowSingletonAction,)):
            n_candidates.append(action.col_j + 1)
        elif isinstance(action, BoundTightenAction):
            n_candidates.append(action.col_j + 1)
    return max(n_candidates)
