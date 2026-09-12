"""
solver/utils/sparse_lu.py
--------------------------
Sparse LU factorization wrapper used by the revised simplex.

Wraps scipy.sparse.linalg.splu to provide:
  - solve(rhs)           →  B⁻¹ rhs   (forward/back substitution)
  - solve_transpose(rhs) →  B⁻ᵀ rhs
  - refactorize(B)       →  recompute LU from a new basis matrix

Design notes
------------
- splu performs a sparse LU with partial pivoting.
- We do NOT yet implement rank-1 Forrest-Tomlin updates; instead we
  re-factorize every REFACTORIZE_EVERY pivots (configurable in config.py).
  The REFACTORIZE_EVERY constant balances factorization cost vs. numerical
  drift accumulation.  The next milestone (Forrest-Tomlin) is tracked in
  master_implementation_plan.md as Ticket 1B-04 stretch goal.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import PIVOT_TOL


class SparseLU:
    """
    Wraps scipy's sparse LU factorization with a clean solve interface.

    Parameters
    ----------
    B : csc_matrix or csr_matrix
        The basis matrix to factorize. Must be square and non-singular.
    """

    def __init__(self, B: sp.spmatrix) -> None:
        self._factor: spla.SuperLU = None  # type: ignore
        self._n: int = B.shape[0]
        self.refactorize(B)

    def refactorize(self, B: sp.spmatrix) -> None:
        """
        Compute a fresh LU factorization of B.
        Should be called every REFACTORIZE_EVERY pivots or when numerical
        errors are detected.
        """
        B_csc = sp.csc_matrix(B)
        try:
            self._factor = spla.splu(B_csc, permc_spec="COLAMD")
        except RuntimeError as e:
            raise ValueError(
                f"SparseLU: factorization failed — basis matrix may be singular. "
                f"Original error: {e}"
            )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve B x = rhs.  Returns x = B⁻¹ rhs."""
        return self._factor.solve(np.asarray(rhs, dtype=float))

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """Solve Bᵀ x = rhs.  Returns x = B⁻ᵀ rhs."""
        return self._factor.solve(np.asarray(rhs, dtype=float), trans="T")

    @property
    def n(self) -> int:
        return self._n
