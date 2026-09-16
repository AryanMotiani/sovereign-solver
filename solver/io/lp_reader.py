"""
solver/io/lp_reader.py
-----------------------
Parser for the LP format (CPLEX/Gurobi/GLPK style).

Handles the subset of LP format most commonly seen in the wild:
  \\Objective
  Minimize (or Maximize)
   obj: c1 x1 + c2 x2 - c3 x3 + ...
  Subject To (or st:)
   c1: a11 x1 + a12 x2 <= b1
   c2: ...
  Bounds
   0 <= x1 <= 10
   x2 free
  General (integer variables)
   x1 x2
  Binary
   x3
  End

Notes:
  - Case-insensitive keywords
  - Inline comments start with '\\'
  - Coefficients can be omitted (default = 1.0 or -1.0)
  - Supports <=, >=, = constraints
  - Free variables (no bound → [-inf, inf])
  - Returns a Problem dataclass (same as MPS reader)

Example:
    from solver.io.lp_reader import read_lp
    prob = read_lp("instances/afiro.lp")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.problem import Problem


# ── Tokenization ──────────────────────────────────────────────────────────────

_KEYWORD_SECTIONS = {
    "minimize", "minimise", "min",
    "maximize", "maximise", "max",
    "subject to", "st", "s.t.", "such that",
    "bounds", "bound",
    "general", "generals", "gen",
    "binary", "binaries", "bin",
    "end",
}

_FLOAT_RE = re.compile(r"[+-]?\s*\d+\.?\d*(?:[eE][+-]?\d+)?")
_TERM_RE = re.compile(
    r"([+-]?\s*\d*\.?\d*(?:[eE][+-]?\d+)?)\s*\*?\s*([a-zA-Z_][a-zA-Z0-9_.]*)"
)


def _strip_comment(line: str) -> str:
    """Remove inline '\\' comment."""
    idx = line.find("\\")
    return line[:idx].strip() if idx >= 0 else line.strip()


def _parse_expr(text: str) -> List[Tuple[float, str]]:
    """
    Parse a linear expression like '3.5 x1 - 2 x2 + x3'
    into [(3.5, 'x1'), (-2.0, 'x2'), (1.0, 'x3')].
    """
    text = text.strip()
    terms = []
    pos = 0
    sign = +1.0

    while pos < len(text):
        # Skip whitespace
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break

        # Detect sign
        if text[pos] in ("+", "-"):
            sign = -1.0 if text[pos] == "-" else +1.0
            pos += 1
            while pos < len(text) and text[pos].isspace():
                pos += 1

        # Coefficient (optional)
        coef = 1.0
        num_match = re.match(r"\d+\.?\d*(?:[eE][+-]?\d+)?", text[pos:])
        if num_match:
            # Check if immediately followed by a variable name — if yes, coef=num
            # If followed by only whitespace then a variable, coef=num
            coef = float(num_match.group())
            pos += len(num_match.group())
            # Skip optional '*' between coef and variable
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos < len(text) and text[pos] == "*":
                pos += 1
                while pos < len(text) and text[pos].isspace():
                    pos += 1

        # Variable name
        var_match = re.match(r"[a-zA-Z_][a-zA-Z0-9_.]*", text[pos:])
        if var_match:
            varname = var_match.group()
            pos += len(var_match.group())
            terms.append((sign * coef, varname))
        else:
            # No variable — skip unexpected character
            pos += 1

    return terms


def _section_tag(line: str) -> Optional[str]:
    """Return the lowercase section keyword if this line starts a new section."""
    low = line.lower().strip()
    for kw in _KEYWORD_SECTIONS:
        if low == kw or low.startswith(kw + " ") or low.startswith(kw + ":"):
            return kw
    return None


# ── Main parser ───────────────────────────────────────────────────────────────

def read_lp(path: str) -> Problem:
    """
    Parse a CPLEX/Gurobi LP file and return a Problem.

    Parameters
    ----------
    path : str | Path
        Path to the .lp file.

    Returns
    -------
    Problem
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LP file not found: {path}")

    with open(path, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # Clean lines
    lines = [_strip_comment(l) for l in raw_lines]
    lines = [l for l in lines if l]

    # ── State machine ─────────────────────────────────────────────────────────
    sense = "min"
    obj_terms: List[Tuple[float, str]] = []
    constraints: List[Tuple[str, List[Tuple[float, str]], str, float]] = []
    # (name, terms, sense ('<=','>=','='), rhs)
    bounds_raw: Dict[str, Tuple[float, float]] = {}  # var → (lb, ub)
    integer_vars: set = set()
    binary_vars: set = set()

    current_section = None
    current_constr_name = ""
    current_constr_expr = ""
    current_constr_sense = ""

    def _flush_constraint():
        nonlocal current_constr_expr, current_constr_name, current_constr_sense
        if not current_constr_expr:
            return
        # Split on the sense symbol
        for sym in ("<=", ">=", "="):
            if sym in current_constr_expr:
                lhs, rhs = current_constr_expr.split(sym, 1)
                terms = _parse_expr(lhs)
                rhs_val = float(rhs.strip())
                constraints.append((current_constr_name, terms, sym, rhs_val))
                current_constr_expr = ""
                current_constr_name = ""
                current_constr_sense = ""
                return
        current_constr_expr = ""
        current_constr_name = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1

        tag = _section_tag(line)
        if tag is not None:
            _flush_constraint()
            if tag in ("minimize", "minimise", "min"):
                current_section = "obj"
                sense = "min"
            elif tag in ("maximize", "maximise", "max"):
                current_section = "obj"
                sense = "max"
            elif tag in ("subject to", "st", "s.t.", "such that"):
                current_section = "st"
            elif tag in ("bounds", "bound"):
                current_section = "bounds"
            elif tag in ("general", "generals", "gen"):
                current_section = "general"
            elif tag in ("binary", "binaries", "bin"):
                current_section = "binary"
            elif tag == "end":
                break
            continue

        if current_section == "obj":
            # May have "name: expr" form
            if ":" in line:
                _, line = line.split(":", 1)
            obj_terms.extend(_parse_expr(line))

        elif current_section == "st":
            # Constraint line: may be "name: expr <= rhs" or just "expr <= rhs"
            has_name = re.match(r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*:", line)
            if has_name:
                _flush_constraint()
                current_constr_name = has_name.group(1)
                line = line[has_name.end():]

            # Check if this line contains a sense symbol (completing the constraint)
            if any(sym in line for sym in ("<=", ">=", "=")):
                current_constr_expr += " " + line
                _flush_constraint()
            else:
                current_constr_expr += " " + line

        elif current_section == "bounds":
            # Forms: "lb <= x <= ub",  "x >= lb", "x free", "x <= ub"
            lo = line.lower().strip()
            if "free" in lo:
                var = re.search(r"[a-zA-Z_][a-zA-Z0-9_.]*", line)
                if var:
                    bounds_raw[var.group()] = (-np.inf, np.inf)
            elif "<=" in lo or ">=" in lo:
                # Try "lb <= x <= ub"
                m = re.match(
                    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*<=\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*<=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
                    lo
                )
                if m:
                    bounds_raw[m.group(2)] = (float(m.group(1)), float(m.group(3)))
                else:
                    # "x <= ub" or "x >= lb"
                    for sym, is_upper in [("<=", True), (">=", False)]:
                        if sym in lo:
                            parts = lo.split(sym)
                            var = re.search(r"[a-zA-Z_][a-zA-Z0-9_.]*", parts[0])
                            if var:
                                vname = var.group()
                                val = float(re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", parts[1]).group())
                                cur_lb, cur_ub = bounds_raw.get(vname, (0.0, np.inf))
                                if is_upper:
                                    bounds_raw[vname] = (cur_lb, val)
                                else:
                                    bounds_raw[vname] = (val, cur_ub)
                            break

        elif current_section == "general":
            for var in line.split():
                integer_vars.add(var)

        elif current_section == "binary":
            for var in line.split():
                binary_vars.add(var)
                bounds_raw[var] = (0.0, 1.0)
                integer_vars.add(var)

    _flush_constraint()

    # ── Collect all variable names ─────────────────────────────────────────────
    all_vars: List[str] = []
    seen: set = set()
    for coef, var in obj_terms:
        if var not in seen:
            all_vars.append(var)
            seen.add(var)
    for _, terms, _, _ in constraints:
        for coef, var in terms:
            if var not in seen:
                all_vars.append(var)
                seen.add(var)
    var_idx = {v: i for i, v in enumerate(all_vars)}
    n = len(all_vars)

    # ── Build objective ────────────────────────────────────────────────────────
    c = np.zeros(n)
    for coef, var in obj_terms:
        c[var_idx[var]] += coef
    if sense == "max":
        # Negate c so that all solvers (which minimize) correctly maximise.
        # The simplex/IPM/PDHG then negate the returned objective, giving the
        # correct maximum value in the original sense.
        c = -c

    # ── Build constraint matrices ──────────────────────────────────────────────
    ub_rows, ub_cols, ub_data = [], [], []
    ub_rhs = []
    eq_rows, eq_cols, eq_data = [], [], []
    eq_rhs = []

    ub_i = 0
    eq_i = 0
    for (cname, terms, sym, rhs) in constraints:
        if sym == "<=":
            for coef, var in terms:
                ub_rows.append(ub_i)
                ub_cols.append(var_idx[var])
                ub_data.append(coef)
            ub_rhs.append(rhs)
            ub_i += 1
        elif sym == ">=":
            # Negate: ax >= b  →  -ax <= -b
            for coef, var in terms:
                ub_rows.append(ub_i)
                ub_cols.append(var_idx[var])
                ub_data.append(-coef)
            ub_rhs.append(-rhs)
            ub_i += 1
        elif sym == "=":
            for coef, var in terms:
                eq_rows.append(eq_i)
                eq_cols.append(var_idx[var])
                eq_data.append(coef)
            eq_rhs.append(rhs)
            eq_i += 1

    A_ub = sp.csr_matrix(
        (ub_data, (ub_rows, ub_cols)), shape=(ub_i, n), dtype=float
    ) if ub_i > 0 else sp.csr_matrix((0, n))

    A_eq = sp.csr_matrix(
        (eq_data, (eq_rows, eq_cols)), shape=(eq_i, n), dtype=float
    ) if eq_i > 0 else sp.csr_matrix((0, n))

    b_ub = np.array(ub_rhs, dtype=float)
    b_eq = np.array(eq_rhs, dtype=float)

    # ── Build bounds ───────────────────────────────────────────────────────────
    lb = np.zeros(n)
    ub_arr = np.full(n, np.inf)
    for var, (lo, hi) in bounds_raw.items():
        if var in var_idx:
            j = var_idx[var]
            lb[j] = lo
            ub_arr[j] = hi

    # ── Integer mask ───────────────────────────────────────────────────────────
    int_mask = np.zeros(n, dtype=bool)
    for var in integer_vars | binary_vars:
        if var in var_idx:
            int_mask[var_idx[var]] = True

    prob = Problem(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        lb=lb,
        ub=ub_arr,
        integer_mask=int_mask,
        sense=sense,
        name=path.stem,
    )
    prob.validate()
    return prob
