"""
solver/utils/sparse_lu.py
--------------------------
Sparse LU factorization wrapper with rank-1 basis updates.

Product-Form-of-Inverse (PFI) approach:
  After each basis change (column swap), we store the eta vector so that:
    B_new = B_old @ E_k
  where E_k is an elementary matrix (identity with one column replaced).

  Accumulated updates: B_current = B_base @ E_1 @ E_2 @ ... @ E_k
  Inverse: B_current⁻¹ = E_k⁻¹ @ ... @ E_1⁻¹ @ B_base⁻¹

FTRAN (solve B x = rhs):
  1. Apply eta inverses: v = E_k⁻¹ ... E_1⁻¹ rhs   (process in reverse)
  2. x = B_base⁻¹ v

Wait - this is the wrong order. Let me reconsider.

Actually:  B_current = B_base with column[leaving] replaced.
  eta = B_base⁻¹ @ entering_col
  B_current = B_base @ E  where E has column[leaving] = eta
  B_current⁻¹ = E⁻¹ @ B_base⁻¹

So for FTRAN:  x = B_current⁻¹ @ rhs = E_k⁻¹ @ ... @ E_1⁻¹ @ B_base⁻¹ @ rhs
  1. First solve with base: w = B_base⁻¹ @ rhs
  2. Then apply E_1⁻¹, E_2⁻¹, ..., E_k⁻¹ in forward order

For BTRAN: x = B_current⁻ᵀ @ rhs = B_base⁻ᵀ @ E_1⁻ᵀ @ ... @ E_k⁻ᵀ @ rhs
  1. Apply E_k⁻ᵀ, ..., E_1⁻ᵀ in reverse order
  2. Then solve with base transpose

References
----------
    Forrest & Tomlin (1972), Mathematical Programming 2:263-278.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solver.config import PIVOT_TOL


class SparseLU:
    """
    Sparse LU factorization with product-form rank-1 updates.

    Parameters
    ----------
    B : csc_matrix or csr_matrix
        The basis matrix to factorize. Must be square and non-singular.
    """

    def __init__(self, B: sp.spmatrix) -> None:
        self._factor: spla.SuperLU = None  # type: ignore
        self._n: int = B.shape[0]
        # Product-form updates: list of (leaving_idx, eta_column)
        # Each eta is B_old⁻¹ @ entering_col at the time of the update.
        # E_i has column[leaving_idx] = eta.
        self._eta_list: list[tuple[int, np.ndarray]] = []
        self.refactorize(B)

    def refactorize(self, B: sp.spmatrix) -> None:
        """
        Compute a fresh LU factorization of B.
        Clears all accumulated PFI updates.
        """
        B_csc = sp.csc_matrix(B)
        try:
            self._factor = spla.splu(B_csc, permc_spec="COLAMD")
        except RuntimeError as e:
            raise ValueError(
                f"SparseLU: factorization failed — basis matrix may be singular. "
                f"Original error: {e}"
            )
        self._eta_list = []

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve B_current x = rhs.  Returns x = B_current⁻¹ rhs.

        B_current⁻¹ = E_k⁻¹ @ ... @ E_1⁻¹ @ B_base⁻¹
        So: x = E_k⁻¹(... E_1⁻¹(B_base⁻¹ rhs))

        1. w = B_base⁻¹ rhs
        2. Apply E_1⁻¹, then E_2⁻¹, ..., then E_k⁻¹  (forward order)
        """
        rhs = np.asarray(rhs, dtype=float)

        # Step 1: Solve with base factorization
        v = self._factor.solve(rhs)

        # Step 2: Apply eta updates in forward order
        for leaving_idx, eta_col in self._eta_list:
            pivot = eta_col[leaving_idx]
            if abs(pivot) < 1e-15:
                continue
            # Apply E⁻¹: the leaving_idx-th component is divided by pivot,
            # then all other components are adjusted.
            val = v[leaving_idx] / pivot
            v -= val * eta_col
            v[leaving_idx] = val

        return v

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve B_currentᵀ x = rhs.  Returns x = B_current⁻ᵀ rhs.

        B_current⁻ᵀ = B_base⁻ᵀ @ E_1⁻ᵀ @ ... @ E_k⁻ᵀ
        So: x = B_base⁻ᵀ(E_1⁻ᵀ(... E_k⁻ᵀ rhs))

        1. Apply E_k⁻ᵀ, then E_{k-1}⁻ᵀ, ..., then E_1⁻ᵀ  (reverse order)
        2. w = B_base⁻ᵀ result
        """
        rhs = np.asarray(rhs, dtype=float)
        v = rhs.copy()

        # Step 1: Apply eta updates in reverse order (transpose of forward)
        for leaving_idx, eta_col in reversed(self._eta_list):
            pivot = eta_col[leaving_idx]
            if abs(pivot) < 1e-15:
                continue
            # Apply E⁻ᵀ: compute dot product with eta column
            s = np.dot(eta_col, v) - eta_col[leaving_idx] * v[leaving_idx]
            v[leaving_idx] = (v[leaving_idx] - s) / pivot

        # Step 2: Solve with base factorization (transpose)
        return self._factor.solve(v, trans="T")

    def update_column(self, leaving_idx: int, entering_col: np.ndarray) -> None:
        """
        Rank-1 basis update: replace basis column at position leaving_idx
        with entering_col.

        Computes eta = B_current⁻¹ @ entering_col and stores it.

        Parameters
        ----------
        leaving_idx : int
            Index within the basis (0..m-1) of the leaving variable.
        entering_col : np.ndarray
            The constraint-matrix column of the entering variable (A[:, enter]).
        """
        # Compute eta = B_current⁻¹ @ entering_col
        eta = self.solve(entering_col)

        if abs(eta[leaving_idx]) < PIVOT_TOL:
            # Pivot too small — skip update, will be fixed at next refactorize
            return

        self._eta_list.append((leaving_idx, eta.copy()))

    @property
    def n_updates(self) -> int:
        """Number of accumulated rank-1 updates since last refactorize."""
        return len(self._eta_list)

    def needs_refactorize(self, threshold: int = 50) -> bool:
        """True if accumulated updates exceed threshold."""
        return len(self._eta_list) >= threshold

    @property
    def n(self) -> int:
        return self._n
