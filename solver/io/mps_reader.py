"""
solver/io/mps_reader.py
------------------------
MPS file format parser → Problem dataclass.

Handles all standard MPS sections:
  NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA

BOUNDS types supported:
  UP (upper), LO (lower), FX (fixed), FR (free: -inf to +inf),
  MI (minus-infinity lower: -inf to 0), PL (plus: 0 to +inf), BV (binary 0/1)

Integer variables detected via MARKER 'INTORG' / 'INTEND' in COLUMNS section.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.problem import Problem


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tokenize(line: str) -> List[str]:
    """Split a line on whitespace, strip inline comments ($)."""
    line = line.split("$")[0].rstrip()
    return line.split()


def _is_section(tokens: List[str], name: str) -> bool:
    return bool(tokens) and tokens[0].upper() == name.upper()


# ── Main parser ───────────────────────────────────────────────────────────────

def read_mps(path: str) -> Problem:
    """
    Parse an MPS file and return a Problem instance.

    Parameters
    ----------
    path : str or Path
        Path to the .mps file.

    Returns
    -------
    Problem
        Fully populated and validated Problem dataclass.

    Raises
    ------
    ValueError
        If the file is malformed or contains unsupported features.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MPS file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # ── State ─────────────────────────────────────────────────────────────────
    problem_name: str = ""
    sense: str = "min"  # MPS is always minimisation by default

    # Row info: name → ('N'|'L'|'G'|'E')
    row_type: Dict[str, str] = {}
    row_order: List[str] = []   # preserves insertion order
    obj_row: Optional[str] = None

    # Column info: name → index
    var_index: Dict[str, int] = {}
    var_names: List[str] = []

    # Sparse entries: list of (row_name, col_idx, value)
    entries: List[Tuple[str, int, float]] = []

    # RHS: row_name → value
    rhs: Dict[str, float] = {}

    # RANGES: row_name → range_value
    ranges: Dict[str, float] = {}

    # BOUNDS: var_idx → (lb, ub)
    bounds_lo: Dict[int, float] = {}
    bounds_hi: Dict[int, float] = {}

    # Integer marker state
    in_integer_block: bool = False
    integer_vars: set = set()  # var indices

    section: str = ""

    for raw_line in lines:
        # Ignore blank lines and comment lines (starting with *)
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("*"):
            continue

        # Section header: starts at column 0 (no leading space)
        if raw_line[0] not in (" ", "\t"):
            tokens = _tokenize(raw_line)
            if not tokens:
                continue
            section = tokens[0].upper()

            if section == "NAME":
                problem_name = tokens[1] if len(tokens) > 1 else ""
            elif section == "ENDATA":
                break
            continue  # move to next line after processing section header

        # ── Data line (leading whitespace) ─────────────────────────────────
        tokens = _tokenize(raw_line)
        if not tokens:
            continue

        if section == "ROWS":
            row_t = tokens[0].upper()
            row_name = tokens[1]
            row_type[row_name] = row_t
            if row_t == "N" and obj_row is None:
                obj_row = row_name   # first free row = objective
            else:
                row_order.append(row_name)

        elif section == "COLUMNS":
            # Handle integer markers — format: NAME  'MARKER'  'INTORG'|'INTEND'
            # tokens[1] is the literal string 'MARKER' (possibly quoted)
            if len(tokens) >= 3 and tokens[1].strip("'").upper() == "MARKER":
                marker = tokens[2].strip("'")
                if marker.upper() == "INTORG":
                    in_integer_block = True
                elif marker.upper() == "INTEND":
                    in_integer_block = False
                continue
            # Skip any other lines that look like marker declarations
            if len(tokens) >= 2 and tokens[1].strip("'") in ("MARKER", "'MARKER'"):
                continue

            # Regular column entries: var_name row1 val1 [row2 val2]
            var_name = tokens[0]
            if var_name not in var_index:
                var_index[var_name] = len(var_names)
                var_names.append(var_name)
            col_idx = var_index[var_name]

            if in_integer_block:
                integer_vars.add(col_idx)

            # Each line can have 1 or 2 (row, value) pairs
            pairs = list(zip(tokens[1::2], tokens[2::2]))
            for row_name, val_str in pairs:
                entries.append((row_name, col_idx, float(val_str)))

        elif section == "RHS":
            # rhs_name row1 val1 [row2 val2]
            pairs = list(zip(tokens[1::2], tokens[2::2]))
            for row_name, val_str in pairs:
                rhs[row_name] = float(val_str)

        elif section == "RANGES":
            pairs = list(zip(tokens[1::2], tokens[2::2]))
            for row_name, val_str in pairs:
                ranges[row_name] = float(val_str)

        elif section == "BOUNDS":
            bound_type = tokens[0].upper()
            # tokens: BND_TYPE  BND_NAME  VAR_NAME  [VALUE]
            if len(tokens) < 3:
                continue
            var_name = tokens[2]
            if var_name not in var_index:
                # Variable mentioned only in BOUNDS (no columns) — add it
                var_index[var_name] = len(var_names)
                var_names.append(var_name)
            col_idx = var_index[var_name]
            val = float(tokens[3]) if len(tokens) > 3 else 0.0

            if bound_type == "UP":
                bounds_hi[col_idx] = val
            elif bound_type == "LO":
                bounds_lo[col_idx] = val
            elif bound_type == "FX":
                bounds_lo[col_idx] = val
                bounds_hi[col_idx] = val
            elif bound_type == "FR":
                bounds_lo[col_idx] = -np.inf
                bounds_hi[col_idx] = np.inf
            elif bound_type == "MI":
                bounds_lo[col_idx] = -np.inf
                # upper stays default (0 or previously set)
            elif bound_type in ("PL", "BV"):
                if bound_type == "BV":
                    bounds_lo[col_idx] = 0.0
                    bounds_hi[col_idx] = 1.0
                    integer_vars.add(col_idx)
            else:
                # Unknown bound type — skip silently (warn in future)
                pass

    # ── Build matrices ────────────────────────────────────────────────────────
    n_vars = len(var_names)
    if n_vars == 0:
        raise ValueError("No variables found in MPS file.")

    # Separate entries into objective vs. constraints
    all_rows_with_constraints = set(row_order)

    # Build objective vector
    c = np.zeros(n_vars)
    # Build constraint rows
    # We will expand RANGES rows into two rows later
    raw_constraint_rows: List[str] = []  # ordered

    # Collect all entries for objective and constraints
    obj_entries: Dict[int, float] = {}
    con_entries: Dict[str, Dict[int, float]] = {r: {} for r in row_order}

    for row_name, col_idx, val in entries:
        if row_name == obj_row:
            obj_entries[col_idx] = obj_entries.get(col_idx, 0.0) + val
        elif row_name in con_entries:
            con_entries[row_name][col_idx] = (
                con_entries[row_name].get(col_idx, 0.0) + val
            )
        # else: entry references an unknown row (e.g. a secondary free row) — skip

    for col_idx, val in obj_entries.items():
        c[col_idx] = val

    # ── Bounds ────────────────────────────────────────────────────────────────
    lb = np.zeros(n_vars)        # MPS default lb = 0
    ub = np.full(n_vars, np.inf) # MPS default ub = +inf

    for col_idx, lo_val in bounds_lo.items():
        lb[col_idx] = lo_val
    for col_idx, hi_val in bounds_hi.items():
        ub[col_idx] = hi_val

    # Integer variables with default bounds [0, inf] → clamp ub to [0, 1] if BV already set
    # (BV already handled above)

    # ── Build constraint matrices ─────────────────────────────────────────────
    # Separate into ≤ (L after converting G) and = (E) rows.
    # G rows become L by negating coefficients and RHS.
    # RANGES rows produce an extra constraint.

    ineq_rows: List[Tuple[str, Dict[int, float], float]] = []  # (row_name, coefs, rhs_val) for ≤
    eq_rows:   List[Tuple[str, Dict[int, float], float]] = []  # for =

    for row_name in row_order:
        rtype = row_type[row_name]
        coefs = con_entries[row_name]
        rhs_val = rhs.get(row_name, 0.0)
        range_val = ranges.get(row_name, None)

        if rtype == "L":
            ineq_rows.append((row_name, coefs, rhs_val))
            if range_val is not None:
                # L row with RANGE: rhs_val - |range_val| ≤ Ax ≤ rhs_val
                # Add lower bound as ≥ (negated L row)
                neg_coefs = {k: -v for k, v in coefs.items()}
                ineq_rows.append((row_name + "_range_lo", neg_coefs, -(rhs_val - abs(range_val))))

        elif rtype == "G":
            # Negate → L row
            neg_coefs = {k: -v for k, v in coefs.items()}
            ineq_rows.append((row_name, neg_coefs, -rhs_val))
            if range_val is not None:
                # G row with RANGE: rhs_val ≤ Ax ≤ rhs_val + |range_val|
                ineq_rows.append((row_name + "_range_hi", coefs, rhs_val + abs(range_val)))

        elif rtype == "E":
            if range_val is not None:
                # E row with RANGE: rhs_val ≤ Ax ≤ rhs_val + |range_val| (or rhs ± range/2)
                ineq_rows.append((row_name + "_eq_hi", coefs, rhs_val + abs(range_val)))
                neg_coefs = {k: -v for k, v in coefs.items()}
                ineq_rows.append((row_name + "_eq_lo", neg_coefs, -rhs_val))
            else:
                eq_rows.append((row_name, coefs, rhs_val))

    # ── Assemble sparse matrices ──────────────────────────────────────────────
    def _build_sparse(rows_data: List[Tuple[str, Dict[int, float], float]]):
        """Build (csr_matrix, rhs_array) from row data list."""
        m = len(rows_data)
        if m == 0:
            return sp.csr_matrix((0, n_vars), dtype=float), np.zeros(0)
        row_ids, col_ids, vals = [], [], []
        rhs_arr = np.zeros(m)
        for i, (_, coefs, rhs_v) in enumerate(rows_data):
            rhs_arr[i] = rhs_v
            for col_idx, val in coefs.items():
                row_ids.append(i)
                col_ids.append(col_idx)
                vals.append(val)
        mat = sp.csr_matrix(
            (vals, (row_ids, col_ids)), shape=(m, n_vars), dtype=float
        )
        return mat, rhs_arr

    A_ub, b_ub = _build_sparse(ineq_rows)
    A_eq, b_eq = _build_sparse(eq_rows)

    # ── Integer mask ─────────────────────────────────────────────────────────
    integer_mask = np.zeros(n_vars, dtype=bool)
    for col_idx in integer_vars:
        integer_mask[col_idx] = True

    # ── Assemble and validate ─────────────────────────────────────────────────
    problem = Problem(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        lb=lb,
        ub=ub,
        integer_mask=integer_mask,
        sense=sense,
        name=problem_name,
    )
    problem.validate()
    return problem
