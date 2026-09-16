"""
tests/test_mps_reader.py
--------------------------
Phase 1A Gate — MPS reader tests (T-01).

Tests:
  1. Hand-crafted toy MPS file → exact array match
  2. Parse Netlib instances: n_vars, n_constraints, c match HiGHS reader
  3. Integer markers (INTORG/INTEND) correctly detected
  4. BOUNDS types: FX, FR, BV, MI handled correctly
  5. No exception on RANGES section
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from solver.io.mps_reader import read_mps

# Directory for test instances
INST_DIR = Path(__file__).parent / "instances"
INST_DIR.mkdir(exist_ok=True)

TOY_MPS = INST_DIR / "toy_3var.mps"


# ── 1. Hand-crafted toy MPS ───────────────────────────────────────────────────

def test_mps_toy_3var_dimensions():
    """Parse toy_3var.mps and check basic dimensions."""
    prob = read_mps(str(TOY_MPS))
    assert prob.n_vars == 2, f"Expected 2 vars, got {prob.n_vars}"
    # Rows: C1 (L), C2 (L)
    assert prob.n_ineq == 2, f"Expected 2 ineq constraints, got {prob.n_ineq}"
    assert prob.n_eq == 0

def test_mps_toy_3var_objective():
    """Objective coefficients should be -3, -5 (minimisation of negated max obj)."""
    prob = read_mps(str(TOY_MPS))
    # MPS file has OBJ = -3*X1 -5*X2 (for maximisation, stored as-is in MPS)
    np.testing.assert_allclose(prob.c, [-3.0, -5.0], atol=1e-12)

def test_mps_toy_3var_rhs():
    prob = read_mps(str(TOY_MPS))
    # C1: X1 <= 4, C2: 3X1+2X2 <= 18
    np.testing.assert_allclose(prob.b_ub, [4.0, 18.0], atol=1e-12)

def test_mps_toy_3var_bounds():
    prob = read_mps(str(TOY_MPS))
    np.testing.assert_allclose(prob.lb, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(prob.ub, [4.0, 6.0], atol=1e-12)

def test_mps_toy_3var_no_integers():
    prob = read_mps(str(TOY_MPS))
    assert not prob.integer_mask.any(), "Toy LP should have no integer vars"

def test_mps_toy_3var_validates():
    prob = read_mps(str(TOY_MPS))
    prob.validate()  # Should not raise


# ── 2. Integer markers ────────────────────────────────────────────────────────

INTORG_MPS_CONTENT = """\
NAME          INTTEST
ROWS
 N  OBJ
 L  C1
COLUMNS
    MARKER   'MARKER'   'INTORG'
    Y1   OBJ  1.0   C1  1.0
    Y2   OBJ  2.0   C1  1.0
    MARKER   'MARKER'   'INTEND'
    X1   OBJ  3.0   C1  1.0
RHS
    RHS  C1  10.0
BOUNDS
 UP BND Y1  5.0
 UP BND Y2  5.0
 UP BND X1  10.0
ENDATA
"""

def test_mps_intorg_intend_markers(tmp_path):
    """Variables inside INTORG/INTEND should be integer-constrained."""
    f = tmp_path / "inttest.mps"
    f.write_text(INTORG_MPS_CONTENT)
    prob = read_mps(str(f))
    assert prob.n_vars == 3
    # Y1 (index 0) and Y2 (index 1) are integer; X1 (index 2) is continuous
    assert prob.integer_mask[0], "Y1 should be integer"
    assert prob.integer_mask[1], "Y2 should be integer"
    assert not prob.integer_mask[2], "X1 should be continuous"


# ── 3. BV (binary) bound type ─────────────────────────────────────────────────

BV_MPS_CONTENT = """\
NAME          BVTEST
ROWS
 N  OBJ
 L  C1
COLUMNS
    Z1   OBJ  5.0   C1  1.0
    Z2   OBJ  3.0   C1  1.0
RHS
    RHS  C1  1.0
BOUNDS
 BV BND Z1
 BV BND Z2
ENDATA
"""

def test_mps_bv_bounds(tmp_path):
    """BV bound type: binary variable with lb=0, ub=1, integer=True."""
    f = tmp_path / "bvtest.mps"
    f.write_text(BV_MPS_CONTENT)
    prob = read_mps(str(f))
    assert prob.integer_mask[0], "Z1 should be integer (BV)"
    assert prob.integer_mask[1], "Z2 should be integer (BV)"
    assert prob.lb[0] == 0.0 and prob.ub[0] == 1.0, "Z1 should be [0,1]"
    assert prob.lb[1] == 0.0 and prob.ub[1] == 1.0, "Z2 should be [0,1]"


# ── 4. FR (free) and FX (fixed) bound types ───────────────────────────────────

BOUNDS_MPS_CONTENT = """\
NAME          BOUNDSTEST
ROWS
 N  OBJ
 E  C1
COLUMNS
    A  OBJ  1.0  C1  1.0
    B  OBJ  1.0  C1  1.0
    C  OBJ  1.0  C1  1.0
RHS
    RHS  C1  10.0
BOUNDS
 FR BND A
 FX BND B  3.0
 MI BND C
ENDATA
"""

def test_mps_fr_fx_mi_bounds(tmp_path):
    f = tmp_path / "boundstest.mps"
    f.write_text(BOUNDS_MPS_CONTENT)
    prob = read_mps(str(f))
    # A: FR → -inf to +inf
    assert np.isinf(prob.lb[0]) and prob.lb[0] < 0, "A lb should be -inf"
    assert np.isinf(prob.ub[0]) and prob.ub[0] > 0, "A ub should be +inf"
    # B: FX → [3, 3]
    assert prob.lb[1] == 3.0 and prob.ub[1] == 3.0, "B should be fixed at 3"
    # C: MI → -inf lower (default upper = +inf)
    assert np.isinf(prob.lb[2]) and prob.lb[2] < 0, "C lb should be -inf"


# ── 5. RANGES section ─────────────────────────────────────────────────────────

RANGES_MPS_CONTENT = """\
NAME          RANGESTEST
ROWS
 N  OBJ
 L  R1
COLUMNS
    X  OBJ  1.0  R1  1.0
RHS
    RHS  R1  10.0
RANGES
    RNG  R1  3.0
BOUNDS
 UP BND X 15.0
ENDATA
"""

def test_mps_ranges_no_exception(tmp_path):
    """RANGES section should parse without exception."""
    f = tmp_path / "rangestest.mps"
    f.write_text(RANGES_MPS_CONTENT)
    prob = read_mps(str(f))
    # L row with range 3: creates two constraints 7 <= X <= 10
    # We expect at least one extra constraint from the range
    assert prob.n_ineq >= 2, "RANGES on L row should produce 2 constraints"
    prob.validate()


# ── 6. Cross-check against HiGHS on a Netlib instance ────────────────────────

NETLIB_AFIRO_URL = "https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/afiro.mps"
AFIRO_PATH = INST_DIR / "afiro.mps"

def _try_download_afiro():
    if not AFIRO_PATH.exists():
        try:
            urllib.request.urlretrieve(NETLIB_AFIRO_URL, AFIRO_PATH)
        except Exception:
            return False
    return AFIRO_PATH.exists()


@pytest.mark.skipif(
    not _try_download_afiro(),
    reason="Could not download afiro.mps (no internet or file missing)"
)
def test_mps_afiro_matches_highspy():
    """
    Parse afiro.mps with our reader and with highspy.
    Assert n_vars, n_constraints, and objective coefficients match (tol 1e-9).
    """
    import highspy

    our_prob = read_mps(str(AFIRO_PATH))

    h = highspy.Highs()
    h.silent()
    h.readModel(str(AFIRO_PATH))

    lp = h.getLp()
    assert our_prob.n_vars == lp.num_col_, (
        f"n_vars mismatch: ours={our_prob.n_vars}, HiGHS={lp.num_col_}"
    )
    assert our_prob.n_constraints == lp.num_row_, (
        f"n_constraints mismatch: ours={our_prob.n_constraints}, HiGHS={lp.num_row_}"
    )
    np.testing.assert_allclose(
        our_prob.c, np.array(lp.col_cost_), atol=1e-9,
        err_msg="Objective coefficient mismatch vs HiGHS"
    )
