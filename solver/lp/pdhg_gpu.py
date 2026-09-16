"""
solver/lp/pdhg_gpu.py
---------------------
GPU-accelerated Restarted Primal-Dual Hybrid Gradient (PDHG / PDLP) for LP.

Ticket 1C-02 & 1C-02b implementation:
- GPU-resident execution: problem matrices and state vectors stay in device memory
  across all iterations without host-device synchronization overhead.
- Supports CUDA (via CuPy or PyTorch), Apple Silicon Metal (via PyTorch MPS),
  and falls back transparently to vectorized CPU execution when GPU hardware is absent.
- Adaptive restarts on ergodic averages with normalized relative KKT stopping criteria.

References:
    Applegate et al. (2021) "Practical Large-Scale Linear Programming using PDLP", NeurIPS 2021.
    Lu et al. (2023) "cuPDLP.jl: A GPU Implementation of restarted PDHG for LP", arXiv:2307.11679.
    Blin et al. (2026) "Batched First-Order Methods for Parallel LP Solving in MIP", arXiv:2601.21990.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from solver.config import (
    FEASIBILITY_TOL,
    MAX_PDHG_ITERS,
    PDHG_RESTART_PERIOD,
    PDHG_TOL,
)
from solver.lp.pdhg import _to_standard_form_pdhg
from solver.lp.simplex_dense import SolveResult
from solver.problem import Problem

logger = logging.getLogger(__name__)


# ── Hardware device discovery ─────────────────────────────────────────────────

def get_available_device() -> str:
    """
    Detect highest-performing compute device available:
    1. 'cuda' : NVIDIA GPU via CuPy or PyTorch CUDA
    2. 'mps'  : Apple Silicon GPU via PyTorch Metal Performance Shaders
    3. 'cpu'  : Vectorized CPU fallback
    """
    # Check CuPy CUDA
    try:
        import cupy as cp
        if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
            return "cuda"
    except (ImportError, Exception):
        pass

    # Check PyTorch CUDA / MPS
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, Exception):
        pass

    return "cpu"


# ── Abstract Device Engine ───────────────────────────────────────────────────

class DeviceEngine:
    """
    Device-resident acceleration engine for PDHG sparse matrix-vector products,
    projections, and vector norm reductions.
    """

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            self.device_type = get_available_device()
        elif device in ("cuda", "mps", "cpu"):
            available = get_available_device()
            if device != "cpu" and device != available:
                logger.warning(
                    f"Requested device {device!r} not available (detected {available!r}). "
                    f"Falling back to {available!r}."
                )
                self.device_type = available
            else:
                self.device_type = device
        else:
            raise ValueError(f"Unknown device: {device!r}. Choose 'auto', 'cuda', 'mps', or 'cpu'.")

        self.backend = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.device_type == "cuda":
            try:
                import cupy as cp
                self.backend = "cupy"
                return
            except ImportError:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.backend = "torch_cuda"
                        return
                except ImportError:
                    pass
        elif self.device_type == "mps":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.backend = "torch_mps"
                    return
            except ImportError:
                pass

        self.device_type = "cpu"
        self.backend = "numpy"

    def prepare_data(
        self,
        c: np.ndarray,
        A: sp.csr_matrix,
        b: np.ndarray,
        tau: np.ndarray,
        sigma: np.ndarray,
    ):
        """Transfer sparse matrix and vectors into resident device format."""
        if self.backend == "cupy":
            import cupy as cp
            import cupyx.scipy.sparse as cp_sp

            A_dev = cp_sp.csr_matrix(A)
            AT_dev = cp_sp.csr_matrix(A.T.tocsr())
            return {
                "c": cp.asarray(c, dtype=cp.float64),
                "A": A_dev,
                "AT": AT_dev,
                "b": cp.asarray(b, dtype=cp.float64),
                "tau": cp.asarray(tau, dtype=cp.float64),
                "sigma": cp.asarray(sigma, dtype=cp.float64),
                "x": cp.zeros(len(c), dtype=cp.float64),
                "y": cp.zeros(len(b), dtype=cp.float64),
                "x_sum": cp.zeros(len(c), dtype=cp.float64),
                "y_sum": cp.zeros(len(b), dtype=cp.float64),
            }

        elif self.backend in ("torch_cuda", "torch_mps"):
            import torch
            dev_str = "cuda" if self.backend == "torch_cuda" else "mps"
            dev = torch.device(dev_str)

            crow = torch.from_numpy(A.indptr).to(torch.int64).to(dev)
            col = torch.from_numpy(A.indices).to(torch.int64).to(dev)
            val = torch.from_numpy(A.data).to(torch.float32 if dev_str == "mps" else torch.float64).to(dev)
            A_torch = torch.sparse_csr_tensor(crow, col, val, size=A.shape, device=dev)

            AT = A.T.tocsr()
            crow_t = torch.from_numpy(AT.indptr).to(torch.int64).to(dev)
            col_t = torch.from_numpy(AT.indices).to(torch.int64).to(dev)
            val_t = torch.from_numpy(AT.data).to(torch.float32 if dev_str == "mps" else torch.float64).to(dev)
            AT_torch = torch.sparse_csr_tensor(crow_t, col_t, val_t, size=AT.shape, device=dev)

            dtype = torch.float32 if dev_str == "mps" else torch.float64
            return {
                "c": torch.from_numpy(c).to(dtype).to(dev),
                "A": A_torch,
                "AT": AT_torch,
                "b": torch.from_numpy(b).to(dtype).to(dev),
                "tau": torch.from_numpy(tau).to(dtype).to(dev),
                "sigma": torch.from_numpy(sigma).to(dtype).to(dev),
                "x": torch.zeros(len(c), dtype=dtype, device=dev),
                "y": torch.zeros(len(b), dtype=dtype, device=dev),
                "x_sum": torch.zeros(len(c), dtype=dtype, device=dev),
                "y_sum": torch.zeros(len(b), dtype=dtype, device=dev),
            }

        else:
            # Vectorized NumPy / SciPy CPU
            return {
                "c": c.copy(),
                "A": A,
                "AT": A.T.tocsr(),
                "b": b.copy(),
                "tau": tau.copy(),
                "sigma": sigma.copy(),
                "x": np.zeros(len(c), dtype=float),
                "y": np.zeros(len(b), dtype=float),
                "x_sum": np.zeros(len(c), dtype=float),
                "y_sum": np.zeros(len(b), dtype=float),
            }

    def iterate_and_average(
        self,
        data: dict,
        window_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Perform one accelerated PDHG iteration on resident device memory."""
        if self.backend == "cupy":
            import cupy as cp

            x = data["x"]
            y = data["y"]
            c = data["c"]
            b = data["b"]
            tau = data["tau"]
            sigma = data["sigma"]
            A = data["A"]
            AT = data["AT"]

            # Primal step: x_{k+1} = max(0, x - tau * (c - Aᵀ y))
            AT_y = AT.dot(y)
            x_new = cp.maximum(0.0, x - tau * (c - AT_y))

            # Extrapolation: x_bar = 2 x_{k+1} - x_k
            x_bar = 2.0 * x_new - x

            # Dual step: y_{k+1} = y + sigma * (b - A x_bar)
            A_xbar = A.dot(x_bar)
            y_new = y + sigma * (b - A_xbar)

            # Ergodic running sum accumulation on device
            data["x_sum"] += x_new
            data["y_sum"] += y_new
            data["x"] = x_new
            data["y"] = y_new
            return data["x"], data["y"], window_len + 1

        elif self.backend in ("torch_cuda", "torch_mps"):
            import torch

            x = data["x"]
            y = data["y"]
            c = data["c"]
            b = data["b"]
            tau = data["tau"]
            sigma = data["sigma"]
            A = data["A"]
            AT = data["AT"]

            # Primal step
            AT_y = torch.mv(AT, y)
            x_new = torch.clamp(x - tau * (c - AT_y), min=0.0)

            # Extrapolation
            x_bar = 2.0 * x_new - x

            # Dual step
            A_xbar = torch.mv(A, x_bar)
            y_new = y + sigma * (b - A_xbar)

            data["x_sum"] += x_new
            data["y_sum"] += y_new
            data["x"] = x_new
            data["y"] = y_new
            return data["x"], data["y"], window_len + 1

        else:
            # NumPy / SciPy CPU
            x = data["x"]
            y = data["y"]
            c = data["c"]
            b = data["b"]
            tau = data["tau"]
            sigma = data["sigma"]
            A = data["A"]
            AT = data["AT"]

            AT_y = AT.dot(y)
            x_new = np.maximum(0.0, x - tau * (c - AT_y))
            x_bar = 2.0 * x_new - x
            A_xbar = A.dot(x_bar)
            y_new = y + sigma * (b - A_xbar)

            data["x_sum"] += x_new
            data["y_sum"] += y_new
            data["x"] = x_new
            data["y"] = y_new
            return data["x"], data["y"], window_len + 1

    def to_host(self, array_or_tensor) -> np.ndarray:
        """Move array from device to host NumPy array."""
        if self.backend == "cupy":
            import cupy as cp
            return cp.asnumpy(array_or_tensor)
        elif self.backend in ("torch_cuda", "torch_mps"):
            return array_or_tensor.detach().cpu().numpy().astype(float)
        else:
            return np.asarray(array_or_tensor)


# ── Step-size computation ──────────────────────────────────────────────────────

def _compute_step_sizes_gpu(A: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-variable primal step sizes (τ_j) and per-constraint dual
    step sizes (σ_i) using diagonal preconditioning (Pock & Chambolle 2011).
    """
    col_norms = np.array(np.abs(A).sum(axis=0)).ravel()
    row_norms = np.array(np.abs(A).sum(axis=1)).ravel()

    col_norms = np.maximum(col_norms, 1e-4)
    row_norms = np.maximum(row_norms, 1e-4)

    tau = 0.999 / col_norms
    sigma = 0.999 / row_norms
    return tau, sigma


# ── High-Level GPU PDHG Solver Entry Point ─────────────────────────────────────

def solve_lp_pdhg_gpu(
    problem: Problem,
    device: str = "auto",
    max_iters: int = MAX_PDHG_ITERS,
    tol: float = PDHG_TOL,
) -> SolveResult:
    """
    Solve an LP using GPU-accelerated restarted PDHG.

    Parameters
    ----------
    problem : Problem
        The optimization problem instance.
    device : str, default 'auto'
        Compute device target: 'auto', 'cuda', 'mps', or 'cpu'.
    max_iters : int
        Maximum number of iterations.
    tol : float
        KKT relative tolerance.

    Returns
    -------
    SolveResult
        Standard solver result with solution vector, objective value, status, and iters.
    """
    c, A, b, lb_orig, n_orig = _to_standard_form_pdhg(problem)
    m, n = A.shape

    if m == 0 or n == 0:
        return SolveResult("optimal", np.zeros(problem.n_vars), 0.0, 0)

    engine = DeviceEngine(device=device)
    tau, sigma = _compute_step_sizes_gpu(A)

    data = engine.prepare_data(c, A, b, tau, sigma)
    AT = A.T.tocsr()

    norm_b = max(1.0, float(np.linalg.norm(b)))
    norm_c = max(1.0, float(np.linalg.norm(c)))

    def _eval_kkt_host(xp: np.ndarray, yp: np.ndarray) -> Tuple[float, float, float, float]:
        # Primal residual: ‖Ax - b‖ / (1 + ‖b‖)
        rp = float(np.linalg.norm(A.dot(xp) - b)) / norm_b
        # Dual slack: s = max(c - Aᵀy, 0), dual residual: ‖c - Aᵀy - s‖ / (1 + ‖c‖)
        A_T_y = AT.dot(yp)
        s = np.maximum(c - A_T_y, 0.0)
        rd = float(np.linalg.norm(c - A_T_y - s)) / norm_c
        # Relative duality gap: |cᵀx - bᵀy| / (1 + |cᵀx| + |bᵀy|)
        p_obj = float(c @ xp)
        d_obj = float(b @ yp)
        gap = abs(p_obj - d_obj) / (1.0 + abs(p_obj) + abs(d_obj))
        max_err = max(rp, rd, gap)
        return max_err, rp, rd, gap

    x_init = engine.to_host(data["x"])
    y_init = engine.to_host(data["y"])
    current_kkt, _, _, _ = _eval_kkt_host(x_init, y_init)

    window_len = 0
    status = "iteration_limit"
    iters = 0
    check_interval = max(10, min(PDHG_RESTART_PERIOD, 100))

    for iters in range(1, max_iters + 1):
        _, _, window_len = engine.iterate_and_average(data, window_len)

        if window_len >= check_interval and window_len % check_interval == 0:
            x_avg = engine.to_host(data["x_sum"]) / window_len
            y_avg = engine.to_host(data["y_sum"]) / window_len

            err, rp, rd, gap = _eval_kkt_host(x_avg, y_avg)

            if err < tol:
                status = "optimal"
                # Update final solution point
                if engine.backend == "cupy":
                    import cupy as cp
                    data["x"] = cp.asarray(x_avg)
                    data["y"] = cp.asarray(y_avg)
                elif engine.backend in ("torch_cuda", "torch_mps"):
                    import torch
                    dev_str = "cuda" if engine.backend == "torch_cuda" else "mps"
                    dtype = torch.float32 if dev_str == "mps" else torch.float64
                    data["x"] = torch.from_numpy(x_avg).to(dtype).to(torch.device(dev_str))
                    data["y"] = torch.from_numpy(y_avg).to(dtype).to(torch.device(dev_str))
                else:
                    data["x"] = x_avg.copy()
                    data["y"] = y_avg.copy()
                break

            # Adaptive restart: sufficient decrease
            if err < 0.5 * current_kkt:
                if engine.backend == "cupy":
                    import cupy as cp
                    data["x"] = cp.asarray(x_avg)
                    data["y"] = cp.asarray(y_avg)
                    data["x_sum"].fill(0.0)
                    data["y_sum"].fill(0.0)
                elif engine.backend in ("torch_cuda", "torch_mps"):
                    import torch
                    dev_str = "cuda" if engine.backend == "torch_cuda" else "mps"
                    dtype = torch.float32 if dev_str == "mps" else torch.float64
                    data["x"] = torch.from_numpy(x_avg).to(dtype).to(torch.device(dev_str))
                    data["y"] = torch.from_numpy(y_avg).to(dtype).to(torch.device(dev_str))
                    data["x_sum"].zero_()
                    data["y_sum"].zero_()
                else:
                    data["x"] = x_avg.copy()
                    data["y"] = y_avg.copy()
                    data["x_sum"].fill(0.0)
                    data["y_sum"].fill(0.0)
                window_len = 0
                current_kkt = err

    # ── Solution Extraction & Post-processing ─────────────────────────────────
    x_final = engine.to_host(data["x"])
    x_shifted = x_final[:n_orig]
    x_orig = x_shifted + lb_orig
    x_orig = np.clip(
        x_orig,
        np.where(np.isfinite(problem.lb), problem.lb, -1e30),
        np.where(np.isfinite(problem.ub), problem.ub, 1e30),
    )

    obj_val = float(problem.c @ x_orig)
    if problem.sense == "max":
        obj_val = -obj_val

    return SolveResult(status, x_orig, obj_val, iters)
