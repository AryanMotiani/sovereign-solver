"""
solver/problem.py
-----------------
Core Problem dataclass representing a mathematical optimization problem
in standard form understood by all solver modules.

Supports LP, MILP, and QP problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class Problem:
    """
    Unified representation of an optimization problem:

        min/max   cᵀx  +  ½ xᵀ P x          (P = None → LP/MILP)
        s.t.      A_ub x  ≤  b_ub             (m_ub inequality rows)
                  A_eq x  =  b_eq             (m_eq equality rows)
                  lb  ≤  x  ≤  ub             (variable bounds)
                  x[integer_mask] ∈ ℤ

    Attributes
    ----------
    c : ndarray, shape (n,)
        Linear objective coefficients.
    A_ub : csr_matrix, shape (m_ub, n)
        Inequality constraint matrix (Ax ≤ b).
    b_ub : ndarray, shape (m_ub,)
        Inequality RHS.
    A_eq : csr_matrix, shape (m_eq, n)
        Equality constraint matrix (Ax = b).
    b_eq : ndarray, shape (m_eq,)
        Equality RHS.
    lb : ndarray, shape (n,)
        Variable lower bounds. Use -np.inf for unbounded below.
    ub : ndarray, shape (n,)
        Variable upper bounds. Use np.inf for unbounded above.
    integer_mask : ndarray of bool, shape (n,)
        True for integer-constrained variables.
    P_qp : csr_matrix or None, shape (n, n)
        PSD quadratic objective matrix for QP. None → LP/MILP.
    sense : str
        'min' (default) or 'max'. Internally the solver always minimises;
        'max' problems have c negated and the returned objective sign flipped.
    name : str
        Optional problem name (from MPS NAME section).
    """

    c: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integer_mask: np.ndarray
    P_qp: Optional[sp.csr_matrix] = None
    sense: str = "min"
    name: str = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return len(self.c)

    @property
    def n_ineq(self) -> int:
        """Number of inequality constraints (A_ub x ≤ b_ub)."""
        return self.A_ub.shape[0]

    @property
    def n_eq(self) -> int:
        """Number of equality constraints (A_eq x = b_eq)."""
        return self.A_eq.shape[0]

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return self.n_ineq + self.n_eq

    @property
    def is_milp(self) -> bool:
        """True if any variable is integer-constrained."""
        return bool(np.any(self.integer_mask))

    @property
    def is_qp(self) -> bool:
        """True if a quadratic objective term P is present."""
        return self.P_qp is not None

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Raise ValueError if any dimension is inconsistent.
        Call this after constructing a Problem (e.g. at end of MPS reader).
        """
        n = self.n_vars
        errors = []

        if self.A_ub.shape != (self.n_ineq, n):
            errors.append(
                f"A_ub shape {self.A_ub.shape} inconsistent with n_vars={n}"
            )
        if self.b_ub.shape != (self.n_ineq,):
            errors.append(f"b_ub shape {self.b_ub.shape} != ({self.n_ineq},)")
        if self.A_eq.shape != (self.n_eq, n):
            errors.append(
                f"A_eq shape {self.A_eq.shape} inconsistent with n_vars={n}"
            )
        if self.b_eq.shape != (self.n_eq,):
            errors.append(f"b_eq shape {self.b_eq.shape} != ({self.n_eq},)")
        if self.lb.shape != (n,):
            errors.append(f"lb shape {self.lb.shape} != ({n},)")
        if self.ub.shape != (n,):
            errors.append(f"ub shape {self.ub.shape} != ({n},)")
        if self.integer_mask.shape != (n,):
            errors.append(
                f"integer_mask shape {self.integer_mask.shape} != ({n},)"
            )
        if self.sense not in ("min", "max"):
            errors.append(f"sense must be 'min' or 'max', got '{self.sense}'")
        if self.P_qp is not None and self.P_qp.shape != (n, n):
            errors.append(
                f"P_qp shape {self.P_qp.shape} inconsistent with n_vars={n}"
            )
        if np.any(self.lb > self.ub + 1e-12):
            bad = np.where(self.lb > self.ub + 1e-12)[0]
            errors.append(f"lb > ub for variables: {bad[:5].tolist()} ...")

        if errors:
            raise ValueError("Problem validation failed:\n  " + "\n  ".join(errors))

    # ── Convenience factory ───────────────────────────────────────────────────

    @classmethod
    def empty(cls, n: int) -> "Problem":
        """Create a trivial n-variable Problem with no constraints (useful in tests)."""
        return cls(
            c=np.zeros(n),
            A_ub=sp.csr_matrix((0, n), dtype=float),
            b_ub=np.zeros(0),
            A_eq=sp.csr_matrix((0, n), dtype=float),
            b_eq=np.zeros(0),
            lb=np.zeros(n),
            ub=np.full(n, np.inf),
            integer_mask=np.zeros(n, dtype=bool),
        )

    def eval_obj(self, x: np.ndarray) -> float:
        """
        Evaluate the objective at point x (in the original sense — max or min).
        """
        val = float(self.c @ x)
        if self.P_qp is not None:
            val += 0.5 * float(x @ self.P_qp.dot(x))
        if self.sense == "max":
            val = -val  # internally we store negated c for max problems
        return val

    def __repr__(self) -> str:
        kind = "MILP" if self.is_milp else ("QP" if self.is_qp else "LP")
        return (
            f"Problem(name={self.name!r}, type={kind}, "
            f"n_vars={self.n_vars}, n_ineq={self.n_ineq}, n_eq={self.n_eq})"
        )
