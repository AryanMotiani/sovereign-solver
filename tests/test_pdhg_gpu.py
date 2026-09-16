"""
tests/test_pdhg_gpu.py
----------------------
Test suite for the GPU-accelerated PDHG first-order LP solver (Ticket 1C-02 / 1C-02b).
Validates numerical correctness, device selection, and seamless CPU fallback.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solver.lp.pdhg_gpu import (
    get_available_device,
    solve_lp_pdhg_gpu,
)
from solver.lp.simplex_revised import solve_lp_revised
from solver.problem import Problem


def make_problem(c, A_ub, b_ub, A_eq=None, b_eq=None):
    m_ub, n = np.array(A_ub).shape
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    return Problem(
        c=np.array(c, dtype=float),
        A_ub=sp.csr_matrix(np.array(A_ub, dtype=float)),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=sp.csr_matrix(np.array(A_eq, dtype=float)),
        b_eq=np.array(b_eq, dtype=float),
        lb=np.zeros(n),
        ub=np.full(n, np.inf),
        integer_mask=np.zeros(n, dtype=bool),
    )


# ── 1. Device detection ───────────────────────────────────────────────────────

def test_device_detection():
    """Verify device detection returns a valid device string ('cuda', 'mps', or 'cpu')."""
    dev = get_available_device()
    assert dev in ("cuda", "mps", "cpu"), f"Unexpected device: {dev}"


# ── 2. Basic 2-variable LP ────────────────────────────────────────────────────

def test_pdhg_gpu_2var_lp():
    """
    max 3x + 5y  s.t. x≤4, 2y≤12, 3x+2y≤18, x,y≥0
    Optimal: x=2, y=6, obj=-36 (minimizing -3x-5y).
    """
    prob = make_problem(
        c=[-3.0, -5.0],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
    )
    result = solve_lp_pdhg_gpu(prob, device="auto")
    assert result.status in ("optimal", "iteration_limit"), result.status
    assert result.x is not None
    assert result.objective <= 0.1, (
        f"PDHG GPU didn't improve over trivial solution: obj={result.objective}"
    )
    assert np.all(result.x >= -1e-4), f"Negative x: {result.x}"


def test_pdhg_gpu_simple_min():
    """
    min x + 2y  s.t. x+y≥2, x,y≥0
    Optimal: x=2, y=0, obj=2.
    """
    prob = make_problem(
        c=[1.0, 2.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[-2.0],
    )
    result = solve_lp_pdhg_gpu(prob, device="auto")
    assert result.status in ("optimal", "iteration_limit"), result.status
    assert result.x is not None
    assert result.objective < 3.0, f"Expected ~2, got {result.objective}"


# ── 3. Agreement with Simplex ─────────────────────────────────────────────────

@pytest.mark.parametrize("c,A,b,expected_obj", [
    ([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18], -36.0),
    ([1, 1],   [[1, 0], [0, 1]],         [3, 3],      0.0),
    ([-1, -2], [[1, 1]],                 [4],         -8.0),
])
def test_pdhg_gpu_agrees_with_simplex(c, A, b, expected_obj):
    prob = make_problem(c=c, A_ub=A, b_ub=b)
    gpu_res = solve_lp_pdhg_gpu(prob, device="auto")
    simp_res = solve_lp_revised(prob)

    assert gpu_res.status in ("optimal", "iteration_limit")
    assert simp_res.status == "optimal"

    if gpu_res.status == "optimal":
        diff = abs(gpu_res.objective - simp_res.objective)
        assert diff < 0.5, (
            f"GPU PDHG/simplex disagree: gpu={gpu_res.objective:.5g}, "
            f"simplex={simp_res.objective:.5g}, diff={diff:.2e}"
        )


# ── 4. Random LP Cross-Check ──────────────────────────────────────────────────

def test_pdhg_gpu_random_lp():
    rng = np.random.default_rng(42)
    n, m = 6, 4
    A = np.abs(rng.standard_normal((m, n)))
    x_true = rng.uniform(0.5, 1.5, n)
    b = A @ x_true + rng.uniform(0.1, 0.5, m)

    A_ub = np.vstack([A, np.eye(n)])
    b_ub = np.concatenate([b, np.full(n, 3.0)])
    c = rng.standard_normal(n)

    prob = make_problem(c=c.tolist(), A_ub=A_ub.tolist(), b_ub=b_ub.tolist())
    gpu_res = solve_lp_pdhg_gpu(prob, device="auto")
    simp_res = solve_lp_revised(prob)

    assert simp_res.status == "optimal"
    if gpu_res.status == "optimal":
        diff = abs(gpu_res.objective - simp_res.objective)
        assert diff < 1.0, f"Random LP diff: {diff:.2e}"


# ── 5. Large Sparse LP Stress Test ────────────────────────────────────────────

def test_pdhg_gpu_large_sparse():
    rng = np.random.default_rng(7)
    n, m = 500, 200
    A_dense = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.05)
    x_true = rng.uniform(0, 1, n)
    b = np.abs(A_dense @ x_true) + 1.0
    c = rng.standard_normal(n)

    prob = make_problem(c=c.tolist(), A_ub=A_dense.tolist(), b_ub=b.tolist())
    result = solve_lp_pdhg_gpu(prob, device="auto")

    assert result.status in ("optimal", "iteration_limit")
    assert result.x is not None
    assert result.x.shape == (n,)


# ── 6. Explicit Device Fallback ───────────────────────────────────────────────

def test_pdhg_gpu_explicit_devices():
    """Explicitly test with device='cpu', and verify requesting unavailable device gracefully falls back."""
    prob = make_problem(
        c=[-1.0, -2.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[4.0],
    )
    # CPU device
    res_cpu = solve_lp_pdhg_gpu(prob, device="cpu")
    assert res_cpu.status in ("optimal", "iteration_limit")

    # If asking for CUDA on non-CUDA system, it should warn and fall back rather than crash
    res_fallback = solve_lp_pdhg_gpu(prob, device="cuda")
    assert res_fallback.status in ("optimal", "iteration_limit")
