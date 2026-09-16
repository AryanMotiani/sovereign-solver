"""
solver/io/qps_reader.py
------------------------
Parser for QPS (Quadratic Programming Standard) format.

QPS extends MPS format with a QUADOBJ section specifying the upper-triangular
entries of the quadratic objective matrix Q in the problem:

    min   (1/2) xᵀQx + cᵀx
    s.t.  Ax ≤ b,  Aeq x = beq,  lb ≤ x ≤ ub

Format structure (MPS sections + one extra):
  NAME          problem_name
  ROWS          — same as MPS (objective + constraints)
  COLUMNS       — same as MPS (linear objective + constraint coefficients)
  RHS           — same as MPS
  BOUNDS        — same as MPS
  QUADOBJ       — upper-triangular Q entries:
                    col_i  col_j  q_ij
  ENDATA

The QUADOBJ section lists entries of Q as:
    col_name_i   col_name_j   value
where value is q_ij (for i≤j). The factor 1/2 is NOT applied by the format;
by convention, diagonal entries q_ii represent the coefficient of (1/2) x_i²,
so we store Q directly and multiply by 1/2 when evaluating the objective.

References:
    Maros & Mészáros (1999) "A repository of convex quadratic programming problems"
    ILOG CPLEX QPS format documentation.

Usage:
    from solver.io.qps_reader import read_qps
    prob = read_qps("instances/qplib/qp_small.qps")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.problem import Problem
from solver.io.mps_reader import read_mps


def read_qps(path: str) -> Problem:
    """
    Parse a QPS file and return a Problem with P_qp set.

    Parameters
    ----------
    path : str | Path
        Path to the .qps file.

    Returns
    -------
    Problem
        Identical to reading the MPS sections, but with P_qp (the quadratic
        objective matrix) populated from the QUADOBJ section.
        If no QUADOBJ section is present, returns a plain LP Problem.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QPS file not found: {path}")

    with open(path, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # ── Phase 1: Extract QUADOBJ section lines and remove them ───────────────
    # We'll parse MPS sections normally and handle QUADOBJ separately.
    mps_lines: List[str] = []
    quad_lines: List[str] = []
    in_quadobj = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            mps_lines.append(line)
            continue
        upper = stripped.upper()
        if upper.startswith("QUADOBJ"):
            in_quadobj = True
            continue
        if in_quadobj:
            # Any new section keyword ends QUADOBJ
            if any(upper.startswith(kw) for kw in ("ROWS", "COLUMNS", "RHS",
                                                     "BOUNDS", "RANGES",
                                                     "ENDATA", "NAME")):
                in_quadobj = False
                mps_lines.append(line)
            else:
                quad_lines.append(stripped)
        else:
            mps_lines.append(line)

    # ── Phase 2: Parse the MPS parts (reuse our MPS reader via tmp file) ─────
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mps", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.writelines(mps_lines)
        tmp_path = tmp.name

    try:
        base_prob = read_mps(tmp_path)
    finally:
        os.unlink(tmp_path)

    # ── Phase 3: Parse QUADOBJ entries ───────────────────────────────────────
    if not quad_lines:
        # No QUADOBJ section → pure LP
        return base_prob

    n = base_prob.n_vars

    # Build variable name → index map from the problem
    # The MPS reader stores variable names in order; we retrieve them
    # by re-reading the variable columns. We use a pragmatic approach:
    # re-parse COLUMNS section from mps_lines to get variable order.
    var_idx = _extract_var_map(mps_lines)

    # Quadratic matrix (symmetric, stored upper triangle in file)
    q_rows, q_cols, q_data = [], [], []
    for line in quad_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        vi, vj = parts[0], parts[1]
        try:
            val = float(parts[2])
        except ValueError:
            continue
        if vi not in var_idx or vj not in var_idx:
            continue
        i, j = var_idx[vi], var_idx[vj]
        q_rows.append(i)
        q_cols.append(j)
        q_data.append(val)
        if i != j:
            # Symmetrise
            q_rows.append(j)
            q_cols.append(i)
            q_data.append(val)

    if not q_rows:
        return base_prob

    P_qp = sp.csr_matrix(
        (q_data, (q_rows, q_cols)), shape=(n, n), dtype=float
    )

    # Return Problem with P_qp attached
    return Problem(
        c=base_prob.c,
        A_ub=base_prob.A_ub,
        b_ub=base_prob.b_ub,
        A_eq=base_prob.A_eq,
        b_eq=base_prob.b_eq,
        lb=base_prob.lb,
        ub=base_prob.ub,
        integer_mask=base_prob.integer_mask,
        P_qp=P_qp,
        sense=base_prob.sense,
        name=base_prob.name,
    )


def _extract_var_map(mps_lines: List[str]) -> Dict[str, int]:
    """
    Extract the variable name → column index mapping from MPS COLUMNS lines.
    Returns dict {var_name: index}.
    """
    var_idx: Dict[str, int] = {}
    idx = 0
    in_columns = False
    in_integer_marker = False

    for line in mps_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        upper = stripped.upper()
        if upper.startswith("COLUMNS"):
            in_columns = True
            continue
        if in_columns:
            if any(upper.startswith(kw) for kw in ("RHS", "BOUNDS", "RANGES",
                                                     "QUADOBJ", "ENDATA")):
                break
            if "MARKER" in upper:
                continue
            parts = stripped.split()
            if len(parts) >= 1:
                vname = parts[0]
                if vname not in var_idx:
                    var_idx[vname] = idx
                    idx += 1
    return var_idx
