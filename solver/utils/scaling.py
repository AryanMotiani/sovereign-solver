"""
solver/utils/scaling.py
------------------------
Matrix scaling / equilibration for numerical robustness.

Implements two strategies:
  1. Ruiz Equilibration (Ruiz 2001)
     - Iteratively scale rows and columns to unit ∞-norm.
     - Converges in ~5-10 iterations for typical LP matrices.
     - Dramatically improves condition number of A and A D Aᵀ.

  2. Geometric Mean Scaling
     - Scale each row / column so that the geometric mean of
       absolute nonzero entries is 1.
     - Simpler and faster than Ruiz, but less effective on
       extremely ill-conditioned matrices.

Usage
-----
    scaler = RuizScaler()
    A_s, b_s, c_s, lb_s, ub_s = scaler.scale(A, b, c, lb, ub)
    # ... solve scaled problem ...
    x_orig = scaler.unscale_x(x_scaled)
    y_orig = scaler.unscale_y(y_scaled)

References
----------
    Ruiz (2001) "A Scaling Algorithm to Equilibrate Both Rows and Columns
      Norms in Matrices", ENSEEIHT-IRIT Technical Report.
    Tomlin (1975) "On scaling linear programming problems",
      Mathematical Programming Study 4.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.sparse as sp


class RuizScaler:
    """
    Ruiz equilibration: iteratively scale rows/columns to unit ∞-norm.

    After scaling:
        A_scaled = D_r @ A @ D_c
        b_scaled = D_r @ b
        c_scaled = D_c @ c
        x_scaled = D_c⁻¹ @ x       (so A_scaled @ x_scaled = D_r @ A @ x = D_r @ b = b_scaled)
        y_scaled = D_r⁻¹ @ y       (dual variables)

    Parameters
    ----------
    max_iters : int
        Maximum Ruiz iterations. Usually 5-10 suffice.
    tol : float
        Stop when max deviation from 1.0 in row/col norms is below tol.
    """

    def __init__(self, max_iters: int = 10, tol: float = 1e-2):
        self.max_iters = max_iters
        self.tol = tol
        self._row_scale: np.ndarray | None = None  # D_r diagonal
        self._col_scale: np.ndarray | None = None  # D_c diagonal
        self._scaled = False

    def scale(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> tuple:
        """
        Scale the LP data.

        Returns
        -------
        (A_s, b_s, c_s, lb_s, ub_s)
        """
        m, n = A.shape
        if m == 0 or n == 0:
            self._row_scale = np.ones(max(m, 1))
            self._col_scale = np.ones(max(n, 1))
            self._scaled = True
            return A, b, c, lb, ub

        # Work with CSC for efficient column operations
        A_work = A.copy().tocsc()

        # Accumulate scaling factors
        row_scale = np.ones(m)
        col_scale = np.ones(n)

        for _ in range(self.max_iters):
            # Row scaling: D_r[i] = 1 / sqrt(max|A[i,:]|)
            A_abs = A_work.copy()
            A_abs.data = np.abs(A_abs.data)

            # Row norms (∞-norm)
            A_csr = A_abs.tocsr()
            row_norms = np.zeros(m)
            for i in range(m):
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i + 1]
                if row_end > row_start:
                    row_norms[i] = A_csr.data[row_start:row_end].max()
                else:
                    row_norms[i] = 1.0

            # Column norms (∞-norm)
            col_norms = np.zeros(n)
            for j in range(n):
                col_start = A_abs.indptr[j]
                col_end = A_abs.indptr[j + 1]
                if col_end > col_start:
                    col_norms[j] = A_abs.data[col_start:col_end].max()
                else:
                    col_norms[j] = 1.0

            # Compute scaling factors (inverse sqrt of norms)
            dr = np.where(row_norms > 1e-15, 1.0 / np.sqrt(row_norms), 1.0)
            dc = np.where(col_norms > 1e-15, 1.0 / np.sqrt(col_norms), 1.0)

            # Apply scaling: A ← diag(dr) @ A @ diag(dc)
            # Row scaling
            A_csr_work = A_work.tocsr()
            for i in range(m):
                start = A_csr_work.indptr[i]
                end = A_csr_work.indptr[i + 1]
                A_csr_work.data[start:end] *= dr[i]
            A_work = A_csr_work.tocsc()

            # Column scaling
            for j in range(n):
                start = A_work.indptr[j]
                end = A_work.indptr[j + 1]
                A_work.data[start:end] *= dc[j]

            row_scale *= dr
            col_scale *= dc

            # Check convergence
            max_dev = max(
                np.max(np.abs(row_norms * dr**2 - 1.0)) if m > 0 else 0.0,
                np.max(np.abs(col_norms * dc**2 - 1.0)) if n > 0 else 0.0,
            )
            if max_dev < self.tol:
                break

        self._row_scale = row_scale
        self._col_scale = col_scale
        self._scaled = True

        # Apply accumulated scaling
        A_s = A_work.tocsr()
        b_s = b * row_scale
        c_s = c * col_scale

        # Scale bounds: lb_s = lb / D_c, ub_s = ub / D_c
        # Since x_scaled = D_c⁻¹ x, bounds on x_scaled are lb/D_c, ub/D_c
        inv_col = np.where(np.abs(col_scale) > 1e-15, 1.0 / col_scale, 1.0)
        lb_s = lb * inv_col
        ub_s = ub * inv_col

        return A_s, b_s, c_s, lb_s, ub_s

    def unscale_x(self, x_scaled: np.ndarray) -> np.ndarray:
        """Recover original x from scaled solution: x = D_c @ x_scaled."""
        if not self._scaled or self._col_scale is None:
            return x_scaled
        n = len(x_scaled)
        cs = self._col_scale[:n] if len(self._col_scale) >= n else np.pad(
            self._col_scale, (0, n - len(self._col_scale)), constant_values=1.0
        )
        return x_scaled * cs

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Recover original dual y from scaled solution: y = D_r @ y_scaled."""
        if not self._scaled or self._row_scale is None:
            return y_scaled
        m = len(y_scaled)
        rs = self._row_scale[:m] if len(self._row_scale) >= m else np.pad(
            self._row_scale, (0, m - len(self._row_scale)), constant_values=1.0
        )
        return y_scaled * rs

    def unscale_obj(self, obj_scaled: float) -> float:
        """Unscale objective value (no change needed for LP)."""
        return obj_scaled


class GeometricMeanScaler:
    """
    Geometric mean scaling: scale rows/columns so geometric mean
    of absolute nonzero entries is 1.

    Simpler and faster than Ruiz. Good default for well-structured problems.
    """

    def __init__(self, max_iters: int = 5):
        self.max_iters = max_iters
        self._row_scale: np.ndarray | None = None
        self._col_scale: np.ndarray | None = None
        self._scaled = False

    def scale(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> tuple:
        m, n = A.shape
        if m == 0 or n == 0:
            self._row_scale = np.ones(max(m, 1))
            self._col_scale = np.ones(max(n, 1))
            self._scaled = True
            return A, b, c, lb, ub

        A_work = A.copy().tocsr()
        row_scale = np.ones(m)
        col_scale = np.ones(n)

        for _ in range(self.max_iters):
            # Row geometric means
            dr = np.ones(m)
            for i in range(m):
                start = A_work.indptr[i]
                end = A_work.indptr[i + 1]
                if end > start:
                    vals = np.abs(A_work.data[start:end])
                    vals = vals[vals > 1e-15]
                    if len(vals) > 0:
                        gm = np.exp(np.mean(np.log(vals)))
                        dr[i] = 1.0 / np.sqrt(gm) if gm > 1e-15 else 1.0

            # Apply row scaling
            for i in range(m):
                start = A_work.indptr[i]
                end = A_work.indptr[i + 1]
                A_work.data[start:end] *= dr[i]
            row_scale *= dr

            # Column geometric means
            A_csc = A_work.tocsc()
            dc = np.ones(n)
            for j in range(n):
                start = A_csc.indptr[j]
                end = A_csc.indptr[j + 1]
                if end > start:
                    vals = np.abs(A_csc.data[start:end])
                    vals = vals[vals > 1e-15]
                    if len(vals) > 0:
                        gm = np.exp(np.mean(np.log(vals)))
                        dc[j] = 1.0 / np.sqrt(gm) if gm > 1e-15 else 1.0

            for j in range(n):
                start = A_csc.indptr[j]
                end = A_csc.indptr[j + 1]
                A_csc.data[start:end] *= dc[j]
            A_work = A_csc.tocsr()
            col_scale *= dc

        self._row_scale = row_scale
        self._col_scale = col_scale
        self._scaled = True

        b_s = b * row_scale
        c_s = c * col_scale
        inv_col = np.where(np.abs(col_scale) > 1e-15, 1.0 / col_scale, 1.0)
        lb_s = lb * inv_col
        ub_s = ub * inv_col

        return A_work, b_s, c_s, lb_s, ub_s

    def unscale_x(self, x_scaled: np.ndarray) -> np.ndarray:
        if not self._scaled or self._col_scale is None:
            return x_scaled
        return x_scaled * self._col_scale[:len(x_scaled)]

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self._scaled or self._row_scale is None:
            return y_scaled
        return y_scaled * self._row_scale[:len(y_scaled)]


def make_scaler(method: str = "ruiz"):
    """Factory for scaler objects."""
    if method == "ruiz":
        return RuizScaler()
    elif method == "geometric":
        return GeometricMeanScaler()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown scaling method: {method!r}. Use 'ruiz', 'geometric', or 'none'.")


def scale_problem(problem: Problem, method: str = "ruiz") -> tuple[Problem, Optional[Any]]:
    """
    Scale a Problem instance using the specified equilibration method.
    Returns (scaled_problem, scaler).
    """
    scaler = make_scaler(method)
    if scaler is None:
        return problem, None

    # Handle inequalities and equalities
    # We combine them temporarily for scaling or scale A_ub
    if problem.n_ineq > 0:
        A_ub_s, b_ub_s, c_s, lb_s, ub_s = scaler.scale(
            problem.A_ub, problem.b_ub, problem.c, problem.lb, problem.ub
        )
        scaled_p = Problem(
            c=c_s,
            A_ub=A_ub_s,
            b_ub=b_ub_s,
            A_eq=problem.A_eq,
            b_eq=problem.b_eq,
            lb=lb_s,
            ub=ub_s,
            integer_mask=problem.integer_mask,
            sense=problem.sense,
            name=problem.name + "_scaled",
        )
        return scaled_p, scaler
    return problem, None


def unscale_solution(x_scaled: np.ndarray, scaler: Optional[Any]) -> np.ndarray:
    """Unscale a primal solution vector."""
    if scaler is None or x_scaled is None:
        return x_scaled
    return scaler.unscale_x(x_scaled)

