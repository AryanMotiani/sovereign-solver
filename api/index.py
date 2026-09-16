"""
api/index.py
------------
Sovereign Solver Web API — FastAPI application.

Endpoints:
  GET  /api/              → Health check + version info
  POST /api/solve         → Solve an LP/MILP/QP problem
  POST /api/solve/lp      → Solve LP (JSON body)
  POST /api/solve/milp    → Solve MILP (JSON body)
  POST /api/solve/qp      → Solve QP via ADMM (JSON body)
  POST /api/demo/{name}   → Run a built-in demo
  GET  /api/demos         → List available demos

Vercel serverless function — handles the entire FastAPI app.
"""

from __future__ import annotations

import sys
import os
import time
import json
import traceback
from typing import Any, Dict, List, Optional

# Ensure solver package is importable on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import numpy as np
import scipy.sparse as sp

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sovereign Solver API",
    description=(
        "India's indigenous LP/MILP/QP optimization engine — "
        "built from scratch for SIH (Smart India Hackathon). "
        "No dependency on CPLEX, Gurobi, HiGHS, or any commercial solver."
    ),
    version="0.3.0-alpha",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web frontend from /web directory (for local dev)
import pathlib
_WEB_DIR = pathlib.Path(__file__).parent.parent / "web"
if _WEB_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class LPRequest(BaseModel):
    """Solve a linear program: min cᵀx  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  lb <= x <= ub."""
    c: List[float] = Field(..., description="Objective coefficients (minimize)")
    A_ub: Optional[List[List[float]]] = Field(None, description="Inequality constraint matrix")
    b_ub: Optional[List[float]] = Field(None, description="Inequality RHS")
    A_eq: Optional[List[List[float]]] = Field(None, description="Equality constraint matrix")
    b_eq: Optional[List[float]] = Field(None, description="Equality RHS")
    lb: Optional[List[float]] = Field(None, description="Lower bounds (default: 0)")
    ub: Optional[List[float]] = Field(None, description="Upper bounds (default: inf)")
    method: Optional[str] = Field("auto", description="simplex | ipm | pdhg | auto")
    sense: Optional[str] = Field("min", description="min | max")

    model_config = {"json_schema_extra": {
        "example": {
            "c": [1.0, 2.0],
            "A_ub": [[-1.0, -1.0]],
            "b_ub": [-3.0],
            "lb": [0.0, 0.0],
            "method": "simplex",
            "sense": "min"
        }
    }}


class MILPRequest(BaseModel):
    """Solve a mixed-integer LP."""
    c: List[float]
    A_ub: Optional[List[List[float]]] = None
    b_ub: Optional[List[float]] = None
    A_eq: Optional[List[List[float]]] = None
    b_eq: Optional[List[float]] = None
    lb: Optional[List[float]] = None
    ub: Optional[List[float]] = None
    integer_vars: Optional[List[int]] = Field(None, description="Indices of integer variables")
    binary_vars: Optional[List[int]] = Field(None, description="Indices of binary (0-1) variables")
    time_limit: Optional[float] = Field(30.0, description="B&B time limit in seconds")
    sense: Optional[str] = Field("min", description="min | max")

    model_config = {"json_schema_extra": {
        "example": {
            "c": [-3.0, -2.0],
            "A_ub": [[1.0, 1.0], [1.0, 0.0]],
            "b_ub": [4.0, 2.5],
            "lb": [0.0, 0.0],
            "binary_vars": [0, 1],
            "sense": "max"
        }
    }}


class QPRequest(BaseModel):
    """Solve a convex QP: min (1/2)xᵀQx + cᵀx."""
    c: List[float]
    Q: List[List[float]] = Field(..., description="Symmetric PSD quadratic matrix")
    A_ub: Optional[List[List[float]]] = None
    b_ub: Optional[List[float]] = None
    A_eq: Optional[List[List[float]]] = None
    b_eq: Optional[List[float]] = None
    lb: Optional[List[float]] = None
    ub: Optional[List[float]] = None


class SolveResult(BaseModel):
    status: str
    objective: Optional[float]
    x: Optional[List[float]]
    iterations: Optional[int]
    nodes: Optional[int]
    lp_relaxation: Optional[float]
    gap: Optional[float]
    wall_time: float
    solver: str
    message: str


# ── Problem builder ───────────────────────────────────────────────────────────

def _build_problem(
    c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
    lb=None, ub=None, integer_mask=None, sense="min", name="api_problem"
):
    from solver.problem import Problem
    n = len(c)
    c_arr = np.array(c, float)

    if A_ub and b_ub:
        A_ub_m = sp.csr_matrix(np.array(A_ub, float))
        b_ub_arr = np.array(b_ub, float)
    else:
        A_ub_m = sp.csr_matrix((0, n))
        b_ub_arr = np.zeros(0)

    if A_eq and b_eq:
        A_eq_m = sp.csr_matrix(np.array(A_eq, float))
        b_eq_arr = np.array(b_eq, float)
    else:
        A_eq_m = sp.csr_matrix((0, n))
        b_eq_arr = np.zeros(0)

    lb_arr = np.array(lb, float) if lb else np.zeros(n)
    ub_arr = np.array([u if u is not None else np.inf for u in ub], float) if ub else np.full(n, np.inf)

    int_mask = np.zeros(n, dtype=bool)
    if integer_mask is not None:
        int_mask = np.array(integer_mask, dtype=bool)

    return Problem(
        c=c_arr, A_ub=A_ub_m, b_ub=b_ub_arr,
        A_eq=A_eq_m, b_eq=b_eq_arr,
        lb=lb_arr, ub=ub_arr,
        integer_mask=int_mask, sense=sense, name=name
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/", tags=["Meta"])
def health_check():
    """Health check — returns version and capabilities."""
    return {
        "name": "Sovereign Solver",
        "version": "0.3.0-alpha",
        "status": "online",
        "built_for": "Smart India Hackathon — Sovereign LP/MILP Solver",
        "capabilities": {
            "lp": ["simplex (primal+dual)", "interior point (Mehrotra)", "PDHG"],
            "milp": ["branch-and-cut", "Gomory cuts", "MIR cuts", "pseudocost branching"],
            "qp": ["ADMM"],
            "formats": ["MPS", "LP", "QPS"],
            "presolve": ["bound tightening", "probing", "singleton elimination"],
        },
        "no_commercial_solver": True,
    }


@app.get("/api/demos", tags=["Demos"])
def list_demos():
    """List available built-in demo problems."""
    return {
        "demos": [
            {
                "name": "crude_blend",
                "description": "Crude oil blending LP — 4 crude grades, 6 quality constraints",
                "type": "LP",
                "vars": 4, "constraints": 6,
            },
            {
                "name": "knapsack",
                "description": "0-1 knapsack MILP — 10 items",
                "type": "MILP",
                "vars": 10, "constraints": 1,
            },
            {
                "name": "transport",
                "description": "Transportation LP — 5 ports, 6 depots (India rail freight)",
                "type": "LP",
                "vars": 30, "constraints": 11,
            },
            {
                "name": "supply_chain",
                "description": "Supply chain network MILP — 3 factories, 5 DCs, 8 customers",
                "type": "MILP",
                "vars": 60, "constraints": 21,
            },
        ]
    }


@app.post("/api/demo/{name}", response_model=SolveResult, tags=["Demos"])
def run_demo(name: str):
    """Run a built-in demo problem and return the result."""
    t0 = time.perf_counter()

    if name == "crude_blend":
        return _run_crude_blend(t0)
    elif name == "knapsack":
        return _run_knapsack(t0)
    elif name == "transport":
        return _run_transport(t0)
    elif name == "supply_chain":
        return _run_supply_chain(t0)
    else:
        raise HTTPException(status_code=404, detail=f"Demo '{name}' not found")


@app.post("/api/solve/lp", response_model=SolveResult, tags=["Solve"])
def solve_lp(req: LPRequest):
    """
    Solve a linear program.

    For small problems (n < 100): uses Revised Simplex.
    For larger problems: automatically selects IPM or PDHG.
    """
    t0 = time.perf_counter()
    try:
        prob = _build_problem(
            req.c, req.A_ub, req.b_ub, req.A_eq, req.b_eq,
            req.lb, req.ub, sense=req.sense or "min"
        )

        n = prob.n_vars
        method = req.method or "auto"
        if method == "auto":
            method = "simplex" if n <= 500 else "ipm"

        if method in ("simplex", "revised"):
            from solver.lp.simplex_revised import solve_lp_revised
            r = solve_lp_revised(prob)
            solver_name = "Revised Simplex"
        elif method == "ipm":
            from solver.lp.interior_point import solve_lp_ipm
            r = solve_lp_ipm(prob)
            solver_name = "Mehrotra IPM"
        elif method == "pdhg":
            from solver.lp.pdhg import solve_lp_pdhg
            r = solve_lp_pdhg(prob)
            solver_name = "Chambolle-Pock PDHG"
        else:
            from solver.lp.simplex_revised import solve_lp_revised
            r = solve_lp_revised(prob)
            solver_name = "Revised Simplex"

        return SolveResult(
            status=r.status,
            objective=float(r.objective) if r.objective is not None else None,
            x=r.x.tolist() if r.x is not None else None,
            iterations=r.iterations,
            nodes=None, lp_relaxation=None, gap=None,
            wall_time=round(time.perf_counter() - t0, 4),
            solver=solver_name,
            message=getattr(r, "message", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.post("/api/solve/milp", response_model=SolveResult, tags=["Solve"])
def solve_milp(req: MILPRequest):
    """Solve a mixed-integer linear program using Branch-and-Cut."""
    t0 = time.perf_counter()
    try:
        n = len(req.c)
        int_mask = np.zeros(n, dtype=bool)
        if req.integer_vars:
            for j in req.integer_vars:
                int_mask[j] = True
        if req.binary_vars:
            for j in req.binary_vars:
                int_mask[j] = True

        ub = req.ub or ([np.inf] * n)
        if req.binary_vars:
            for j in req.binary_vars:
                ub[j] = 1.0

        prob = _build_problem(
            req.c, req.A_ub, req.b_ub, req.A_eq, req.b_eq,
            req.lb, ub, integer_mask=int_mask,
            sense=req.sense or "min"
        )

        from solver.milp.branch_and_bound import solve_milp as _solve_milp
        r = _solve_milp(prob, time_limit=req.time_limit or 30.0, verbose=False)

        return SolveResult(
            status=r.status,
            objective=float(r.objective) if r.objective is not None else None,
            x=r.x.tolist() if r.x is not None else None,
            iterations=None,
            nodes=r.nodes,
            lp_relaxation=float(r.lp_relaxation) if r.lp_relaxation is not None else None,
            gap=float(r.gap) if r.gap is not None else None,
            wall_time=round(time.perf_counter() - t0, 4),
            solver="Branch-and-Cut (Gomory + pseudocost)",
            message=getattr(r, "message", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.post("/api/solve/qp", response_model=SolveResult, tags=["Solve"])
def solve_qp(req: QPRequest):
    """Solve a convex QP: min (1/2)xᵀQx + cᵀx using ADMM."""
    t0 = time.perf_counter()
    try:
        n = len(req.c)
        prob = _build_problem(
            req.c, req.A_ub, req.b_ub, req.A_eq, req.b_eq,
            req.lb, req.ub
        )
        Q = sp.csr_matrix(np.array(req.Q, float))

        from solver.qp.admm import solve_qp_admm
        r = solve_qp_admm(prob, P=Q)

        return SolveResult(
            status=r.status,
            objective=float(r.objective) if r.objective is not None else None,
            x=r.x.tolist() if r.x is not None else None,
            iterations=r.iterations,
            nodes=None, lp_relaxation=None, gap=None,
            wall_time=round(time.perf_counter() - t0, 4),
            solver="ADMM (Alternating Direction Method of Multipliers)",
            message="",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


# ── Demo runners ──────────────────────────────────────────────────────────────

def _run_crude_blend(t0):
    from solver.problem import Problem
    from solver.lp.simplex_revised import solve_lp_revised
    # 4 crude grades blended, minimize cost, meet quality specs
    c = np.array([38.0, 42.0, 35.0, 46.0])
    A_ub = np.array([
        [1, 1, 1, 1],        # total <= 100k barrels
        [-0.3, -0.4, -0.2, -0.5],  # sulfur limit
        [0.7, 0.6, 0.8, 0.5],      # gasoline yield
        [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
    ])
    b_ub = np.array([100.0, -20.0, 60.0, -10.0, -15.0, -5.0])
    prob = Problem(
        c=c, A_ub=sp.csr_matrix(A_ub), b_ub=b_ub,
        A_eq=sp.csr_matrix((0, 4)), b_eq=np.zeros(0),
        lb=np.zeros(4), ub=np.full(4, np.inf),
        integer_mask=np.zeros(4, dtype=bool), name="crude_blend",
    )
    r = solve_lp_revised(prob)
    return SolveResult(
        status=r.status, objective=float(r.objective) if r.objective else None,
        x=r.x.tolist() if r.x is not None else None,
        iterations=r.iterations, nodes=None, lp_relaxation=None, gap=None,
        wall_time=round(time.perf_counter() - t0, 4),
        solver="Revised Simplex", message="Crude oil blending LP",
    )


def _run_knapsack(t0):
    from solver.milp.branch_and_bound import solve_milp as _solve_milp
    rng = np.random.default_rng(42)
    n = 10
    w = rng.integers(1, 20, n).astype(float)
    v = rng.integers(5, 30, n).astype(float)
    W = 0.4 * w.sum()
    prob = _build_problem(
        c=(-v).tolist(),
        A_ub=[w.tolist()], b_ub=[W],
        lb=[0.0]*n, ub=[1.0]*n,
        integer_mask=np.ones(n, dtype=bool),
    )
    r = _solve_milp(prob, time_limit=30.0, verbose=False)
    return SolveResult(
        status=r.status, objective=-float(r.objective) if r.objective else None,
        x=r.x.tolist() if r.x is not None else None,
        iterations=None, nodes=r.nodes,
        lp_relaxation=float(r.lp_relaxation) if r.lp_relaxation else None,
        gap=float(r.gap) if r.gap else None,
        wall_time=round(time.perf_counter() - t0, 4),
        solver="Branch-and-Cut", message="0-1 Knapsack (max value shown)",
    )


def _run_transport(t0):
    from demos.transportation_lp import (
        build_transportation_problem, SUPPLY, DEMAND, COST, PORTS, DEPOTS
    )
    from solver.lp.simplex_revised import solve_lp_revised
    prob = build_transportation_problem(SUPPLY, DEMAND, COST, PORTS, DEPOTS)
    r = solve_lp_revised(prob)
    return SolveResult(
        status=r.status,
        objective=float(r.objective) if r.objective is not None else None,
        x=r.x.tolist() if r.x is not None else None,
        iterations=r.iterations, nodes=None, lp_relaxation=None, gap=None,
        wall_time=round(time.perf_counter() - t0, 4),
        solver="Revised Simplex",
        message="India rail freight: 5 ports -> 6 inland depots",
    )


def _run_supply_chain(t0):
    from demos.supply_chain_milp import build_supply_chain_problem
    from solver.milp.branch_and_bound import solve_milp as _solve_milp
    prob = build_supply_chain_problem()
    r = _solve_milp(prob, time_limit=30.0, verbose=False)
    return SolveResult(
        status=r.status,
        objective=float(r.objective) if r.objective is not None else None,
        x=r.x.tolist() if r.x is not None else None,
        iterations=None, nodes=r.nodes,
        lp_relaxation=float(r.lp_relaxation) if r.lp_relaxation is not None else None,
        gap=float(r.gap) if r.gap is not None else None,
        wall_time=round(time.perf_counter() - t0, 4),
        solver="Branch-and-Cut",
        message="Supply chain: 3 factories, 5 DCs, 8 Indian cities",
    )


# ── Vercel handler ────────────────────────────────────────────────────────────
# This makes the FastAPI app work as a Vercel serverless function.
from mangum import Mangum
handler = Mangum(app, lifespan="off")
