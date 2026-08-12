"""Capacitive-billing laws of the 1T1R kernel array, one per scan organization.

The main witness is a hand-written tile whose cells conduct nothing at all:
the two rail ladders then sit exactly at their clamp references, so every
node voltage in the tile is known in closed form and the whole billed total
can be written out by hand — all four node coefficients and both rails carry
a distinct value, so a swapped term or a swapped rail moves the total.

Laws pinned here:

  * ``wl_in_bl_scan`` bills every node's full excursion from ground, with no
    rest level and no establishment concept,
  * ``bl_in_wl_scan`` bills only the displacement away from the ideal held
    boundary, plus the hold's own establishment spread over one row scan,
  * the internal access node rests at its own bit-line level, not at the
    off-state divider level its branch would settle to,
  * a tile already at its declared rest bills no displacement at all, and a
    rest state at ground costs nothing to establish — so at a grounded
    boundary the two organizations bill one and the same ledger,
  * the amortization is exact: ``row_num`` idle scanned solves bill exactly
    the one establishment a single grounded solve of the same rest state
    costs,
  * the mode selects the billing and nothing else — the solved port state is
    bit-identical across modes,
  * a node cap is billed at the displacement of its own node, not at the
    boundary level the line it hangs on is driven from.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled; tiny CPU shapes throughout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox import Profiler
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rOperationMode,
    XbarArray1t1rPolicy,
)
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig, SolverProber

_DTYPE = torch.float64
_COL_NUM = 3
_ROW_NUM = 2

# Two supplies, deliberately unequal: a term billed on the wrong rail moves.
_V_DD_WL__V = 1.1
_V_DD_BL__V = 0.7

# Rail references, both nonzero and unequal.
_BL_V_REF__V = 0.30
_SL_V_REF__V = 0.05

# Per-node totals of the lattice, all four distinct.
_BL_NODE_C__fF = 0.31
_X_NODE_C__fF = 0.37
_SL_NODE_C__fF = 0.41
_WL_NODE_C__fF = 0.43

# Two weight states, distinct divider ratios in either branch.
_VX_RATIO_ON_TABLE = (0.25, 0.75)
_VX_RATIO_OFF_TABLE = (0.10, 0.40)
# The ideal limit of the same off branch: an access device orders of
# magnitude below the storage element leaves the whole drop across itself, so
# the access node sits exactly on the bit line — the rest state the billing
# declares.
_VX_RATIO_OFF_AT_REST = (0.0, 0.0)
_V_WL_ON_THRESHOLD__V = 0.5

# Level one cell's access node rests at, as a function of its grid position.
RestLevel = Callable[[int, int], float]


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Hand-built witness pieces
# ---------------------------------------------------------------------------


def _cell_config(
    *,
    g_cell_on__uS: float,
    vx_ratio_off_table: tuple[float, ...] = _VX_RATIO_OFF_TABLE,
) -> XbarCell1t1rLinearConfig:
    """Linear cell whose WL-off branch is cut off; the on branch is a knob."""
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=(0.0, 0.0),
        g_cell_on_table__uS=(g_cell_on__uS, g_cell_on__uS),
        vx_ratio_on_table=_VX_RATIO_ON_TABLE,
        vx_ratio_off_table=vx_ratio_off_table,
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
    )


def _array_config(
    *,
    g_cell_on__uS: float = 0.0,
    bl_node_c__fF: float = _BL_NODE_C__fF,
    sl_node_c__fF: float = _SL_NODE_C__fF,
    vx_ratio_off_table: tuple[float, ...] = _VX_RATIO_OFF_TABLE,
) -> XbarArray1t1rConfig:
    return XbarArray1t1rConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=1.0e-3,
        sl_segment_r__MOhm=2.0e-3,
        bl_node_c__fF=bl_node_c__fF,
        x_node_c__fF=_X_NODE_C__fF,
        sl_node_c__fF=sl_node_c__fF,
        wl_node_c__fF=_WL_NODE_C__fF,
        cell_config=_cell_config(g_cell_on__uS=g_cell_on__uS, vx_ratio_off_table=vx_ratio_off_table),
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
    )


def _states() -> Tensor:
    """Mixed programmed states over the physical ``(col, row)`` grid."""
    return torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.long)


def _v_wl_grid() -> Tensor:
    """Per-gate WL drive: column-varying, row 0 above threshold and row 1 below.

    Deliberately NOT uniform across the columns, so the WL wire ladder is
    billed per column rather than through one folded row coefficient.
    """
    return torch.tensor([[0.90, 0.00], [0.80, 0.10], [0.70, 0.20]], dtype=_DTYPE)


def _build_array(
    mode: XbarArray1t1rOperationMode,
    *,
    g_cell_on__uS: float = 0.0,
    bl_node_c__fF: float = _BL_NODE_C__fF,
    sl_node_c__fF: float = _SL_NODE_C__fF,
    vx_ratio_off_table: tuple[float, ...] = _VX_RATIO_OFF_TABLE,
) -> XbarArray1t1r:
    array = XbarArray1t1r(
        config=_array_config(
            g_cell_on__uS=g_cell_on__uS,
            bl_node_c__fF=bl_node_c__fF,
            sl_node_c__fF=sl_node_c__fF,
            vx_ratio_off_table=vx_ratio_off_table,
        ),
        policy=XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=0),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        operation_mode=mode,
        v_dd_wl__V=_V_DD_WL__V,
        v_dd_bl__V=_V_DD_BL__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    array.program(_states())
    return array


def _ideal_driver() -> VoltageDriver:
    """Boundary clamp with ``r_out = 0`` and every nonideality off."""
    driver = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(_COL_NUM,),
        dtype=_DTYPE,
        T__K=300.0,
    )
    driver.eval()
    driver.fabricate()
    return driver


def _solve(
    array: XbarArray1t1r,
    v_wl: Tensor,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bl_ref__V: float = _BL_V_REF__V,
    sl_ref__V: float = _SL_V_REF__V,
) -> tuple[Tensor, Tensor]:
    """Settle one array against two ideal clamps; return ``(billed, i_bl_port)``."""
    bl_driver = _ideal_driver()
    sl_driver = _ideal_driver()
    bl_ref = torch.full((_COL_NUM,), bl_ref__V, dtype=_DTYPE)
    sl_ref = torch.full((_COL_NUM,), sl_ref__V, dtype=_DTYPE)
    billed: list[Tensor] = []
    monkeypatch.setattr(array, "_record_dynamic_energy", billed.append)
    with Profiler(), torch.no_grad():
        steady = array.solve_array(
            v_wl,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver.snapshot(v_ref__V=bl_ref, shape=bl_ref.shape),
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver.snapshot(v_ref__V=sl_ref, shape=sl_ref.shape),
        )
    [energy__fJ] = billed
    return energy__fJ, steady.i_bl_port__uA


# ---------------------------------------------------------------------------
# Closed-form expectations for the non-conducting witness
# ---------------------------------------------------------------------------


def _vx_ratio(col: int, row: int, v_wl: Tensor) -> float:
    """Divider ratio the cell's own gate voltage selects."""
    state = int(_states()[col, row])
    on = float(v_wl[col, row]) > _V_WL_ON_THRESHOLD__V
    return _VX_RATIO_ON_TABLE[state] if on else _VX_RATIO_OFF_TABLE[state]


def _v_x(col: int, row: int, v_wl: Tensor) -> float:
    """Access-node voltage of the non-conducting witness at the clamp levels."""
    ratio = _vx_ratio(col, row, v_wl)
    return _BL_V_REF__V - ratio * (_BL_V_REF__V - _SL_V_REF__V)


def _v_x_rest_declared(col: int, row: int) -> float:
    """The declared rest level of the access node: its own bit-line boundary."""
    del col, row
    return _BL_V_REF__V


def _v_x_rest_off_divider(col: int, row: int) -> float:
    """Where the off branch's own divider would put the access node.

    Not the rest level the billing declares — pinned only as the rejected
    alternative, so a ledger that solved the off divider instead of taking
    the bit-line boundary is caught.
    """
    ratio = _VX_RATIO_OFF_TABLE[int(_states()[col, row])]
    return _BL_V_REF__V - ratio * (_BL_V_REF__V - _SL_V_REF__V)


def _expected_wl_in_bl_scan__fJ(v_wl: Tensor) -> float:
    """Every node's full excursion from ground, billed once."""
    total = 0.0
    for col in range(_COL_NUM):
        for row in range(_ROW_NUM):
            drive = abs(float(v_wl[col, row]))
            total += _V_DD_BL__V * _BL_NODE_C__fF * _BL_V_REF__V
            total += _V_DD_BL__V * _X_NODE_C__fF * abs(_v_x(col, row, v_wl))
            total += _V_DD_BL__V * _SL_NODE_C__fF * _SL_V_REF__V
            total += _V_DD_WL__V * _WL_NODE_C__fF * drive
    return total


def _expected_wl_side__fJ(v_wl: Tensor) -> float:
    """The control side alone: each cell's word-line node at its own gate level."""
    total = 0.0
    for col in range(_COL_NUM):
        for row in range(_ROW_NUM):
            total += _V_DD_WL__V * _WL_NODE_C__fF * abs(float(v_wl[col, row]))
    return total


def _expected_rest__fJ(v_x_rest: RestLevel = _v_x_rest_declared) -> float:
    """Establishing the whole rest state from ground; the WL rests at 0 V.

    The access node rests on the bit line, so its cap is charged to the same
    level the BL node's is.
    """
    total = 0.0
    for col in range(_COL_NUM):
        for row in range(_ROW_NUM):
            total += _V_DD_BL__V * _BL_NODE_C__fF * _BL_V_REF__V
            total += _V_DD_BL__V * _X_NODE_C__fF * abs(v_x_rest(col, row))
            total += _V_DD_BL__V * _SL_NODE_C__fF * _SL_V_REF__V
    return total


def _expected_bl_in_wl_scan__fJ(v_wl: Tensor, *, v_x_rest: RestLevel = _v_x_rest_declared) -> float:
    """Displacement away from the held boundary, plus the amortized hold."""
    total = 0.0
    for col in range(_COL_NUM):
        for row in range(_ROW_NUM):
            # Both rail nodes sit exactly at their held level: no term.
            total += _V_DD_BL__V * _X_NODE_C__fF * abs(_v_x(col, row, v_wl) - v_x_rest(col, row))
            total += _V_DD_WL__V * _WL_NODE_C__fF * abs(float(v_wl[col, row]))
    return total + _expected_rest__fJ(v_x_rest) / _ROW_NUM


# ---------------------------------------------------------------------------
# Per-mode closed forms
# ---------------------------------------------------------------------------


def test_wl_in_bl_scan_bills_the_full_excursion_from_ground(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scanned-BL total is the hand-written sum over every cap in the tile."""
    array = _build_array(XbarArray1t1rOperationMode.WL_IN_BL_SCAN)
    v_wl = _v_wl_grid()

    billed, _i_bl = _solve(array, v_wl, monkeypatch)

    assert float(billed) == pytest.approx(_expected_wl_in_bl_scan__fJ(v_wl), rel=1e-12)


def test_bl_in_wl_scan_bills_the_displacement_plus_the_amortized_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scanned-WL total is the hand-written displacement sum plus one row scan's share."""
    array = _build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN)
    v_wl = _v_wl_grid()

    billed, _i_bl = _solve(array, v_wl, monkeypatch)

    assert float(billed) == pytest.approx(_expected_bl_in_wl_scan__fJ(v_wl), rel=1e-12)


# ---------------------------------------------------------------------------
# Where the access node rests
# ---------------------------------------------------------------------------


def test_the_access_node_rests_at_its_bit_line_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rest boundary is declared, so the internal node takes the bit line.

    The witness's off branch holds a nonzero share of the terminal drop, so
    the level the off divider would settle to differs from the bit line at
    every cell: the two candidate rest levels give different totals and the
    billed one is the declared boundary.
    """
    array = _build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN)
    v_wl = _v_wl_grid()

    billed, _i_bl = _solve(array, v_wl, monkeypatch)

    assert float(billed) == pytest.approx(_expected_bl_in_wl_scan__fJ(v_wl), rel=1e-12)
    assert float(billed) != pytest.approx(_expected_bl_in_wl_scan__fJ(v_wl, v_x_rest=_v_x_rest_off_divider), rel=1e-12)


def test_a_tile_at_its_declared_rest_bills_no_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every gate idle and the off branch parked on the bit line, only the hold is billed.

    All four nodes of every cell then sit exactly where the rest boundary
    declares them, so the whole
    displacement ledger vanishes and the access bills its amortized share of
    establishing the hold and nothing else.
    """
    array = _build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN, vx_ratio_off_table=_VX_RATIO_OFF_AT_REST)
    idle = torch.zeros(_COL_NUM, _ROW_NUM, dtype=_DTYPE)

    billed, _i_bl = _solve(array, idle, monkeypatch)

    assert float(billed) == pytest.approx(_expected_rest__fJ() / _ROW_NUM, rel=1e-12)


def test_a_grounded_boundary_makes_the_two_ledgers_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rest state at ground costs nothing to establish, so the modes agree.

    Both boundaries clamped to 0 V put the whole conduction path at the same
    level in either organization: the held ledger's displacement terms are
    measured from zero exactly as the scanned one's are, and its
    establishment term is zero. What is left in both is the control side.
    """
    v_wl = _v_wl_grid()

    scanned_bl, _i_bl = _solve(
        _build_array(XbarArray1t1rOperationMode.WL_IN_BL_SCAN), v_wl, monkeypatch, bl_ref__V=0.0, sl_ref__V=0.0
    )
    scanned_wl, _i_wl = _solve(
        _build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN), v_wl, monkeypatch, bl_ref__V=0.0, sl_ref__V=0.0
    )

    assert float(scanned_bl) == pytest.approx(_expected_wl_side__fJ(v_wl), rel=1e-12)
    assert float(scanned_wl) == pytest.approx(float(scanned_bl), rel=1e-12)


# ---------------------------------------------------------------------------
# Amortization identity
# ---------------------------------------------------------------------------


def test_a_full_row_scan_bills_exactly_one_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """``row_num`` idle scanned solves cost exactly one establishment of the rest state.

    With every gate at 0 V and the off branch parked on the bit line, the
    tile already sits at its rest boundary, so a ``bl_in_wl_scan`` solve
    carries no displacement at all and bills its amortized share alone; the
    same rest state reached from ground is what a ``wl_in_bl_scan`` solve of
    the identical tile bills in full. The scan contract is that identity: one
    hold covers exactly ``row_num`` accesses.
    """
    idle = torch.zeros(_COL_NUM, _ROW_NUM, dtype=_DTYPE)
    held = _build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN, vx_ratio_off_table=_VX_RATIO_OFF_AT_REST)
    grounded = _build_array(XbarArray1t1rOperationMode.WL_IN_BL_SCAN, vx_ratio_off_table=_VX_RATIO_OFF_AT_REST)

    scanned, _i_scanned = _solve(held, idle, monkeypatch)
    from_ground, _i_ground = _solve(grounded, idle, monkeypatch)

    assert float(from_ground) > 0.0
    assert float(from_ground) == pytest.approx(_expected_rest__fJ(), rel=1e-12)
    assert _ROW_NUM * float(scanned) == pytest.approx(float(from_ground), rel=1e-12)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


def test_mode_moves_the_billing_and_leaves_the_solve_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two modes solve one identical network and disagree only on the bill."""
    v_wl = _v_wl_grid()

    e_wl_in_bl, i_wl_in_bl = _solve(_build_array(XbarArray1t1rOperationMode.WL_IN_BL_SCAN), v_wl, monkeypatch)
    e_bl_in_wl, i_bl_in_wl = _solve(_build_array(XbarArray1t1rOperationMode.BL_IN_WL_SCAN), v_wl, monkeypatch)

    assert torch.equal(i_wl_in_bl, i_bl_in_wl)
    assert float(e_wl_in_bl) != float(e_bl_in_wl)
    # A held boundary is never re-charged per solve, so it bills strictly less.
    assert float(e_bl_in_wl) < float(e_wl_in_bl)


# ---------------------------------------------------------------------------
# Where a node cap is billed
# ---------------------------------------------------------------------------


def test_the_bl_node_cap_bills_at_every_cell_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node total hangs on its own node, so it bills that node's displacement.

    Cap values do not enter the DC solve, so raising ``bl_node_c__fF`` alone
    moves the bill by exactly ``v_dd_bl * delta_c`` times the summed node
    displacement — here the whole bit-line node grid, the scanned mode
    resting at ground. Under a conducting tile the clamp the column is driven
    from differs from every node behind it, so the slope separates the node
    law from a bill taken at the boundary.
    """
    delta__fF = 0.5
    v_wl = _v_wl_grid()
    mode = XbarArray1t1rOperationMode.WL_IN_BL_SCAN

    base, _i_base = _solve(_build_array(mode, g_cell_on__uS=80.0), v_wl, monkeypatch)
    with SolverProber() as probe:
        raised, _i_raised = _solve(
            _build_array(mode, g_cell_on__uS=80.0, bl_node_c__fF=_BL_NODE_C__fF + delta__fF),
            v_wl,
            monkeypatch,
        )
    dcop = probe.records[-1].dcop

    # Shape: [col_num, row_num]
    v_node = dcop.v_bl_node
    # Shape: [col_num]
    v_clamp = dcop.v_bl_clamp
    assert float((v_clamp.unsqueeze(-1) - v_node).abs().min()) > 0.0, (
        "witness must draw current, else node and boundary coincide"
    )

    slope__fJ = float(raised) - float(base)
    at_node = _V_DD_BL__V * delta__fF * float(v_node.abs().sum())
    at_clamp = _V_DD_BL__V * delta__fF * _ROW_NUM * float(v_clamp.abs().sum())

    assert slope__fJ == pytest.approx(at_node, rel=1e-10)
    assert slope__fJ != pytest.approx(at_clamp, rel=1e-10)


def test_the_sl_node_cap_bills_at_every_cell_node_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The node law is pinned on each rail by its own slope probe.

    A BL-side probe cannot see the source line: a non-conducting witness
    holds every SL node at the drive level, and a BL-only slope cancels any
    SL term between base and raised. Under the conducting witness the SL
    drive and the nodes behind it differ, so raising ``sl_node_c__fF`` alone
    separates the node law from a bill taken at the drive.
    """
    delta__fF = 0.5
    v_wl = _v_wl_grid()
    mode = XbarArray1t1rOperationMode.WL_IN_BL_SCAN

    base, _i_base = _solve(_build_array(mode, g_cell_on__uS=80.0), v_wl, monkeypatch)
    with SolverProber() as probe:
        raised, _i_raised = _solve(
            _build_array(mode, g_cell_on__uS=80.0, sl_node_c__fF=_SL_NODE_C__fF + delta__fF),
            v_wl,
            monkeypatch,
        )
    dcop = probe.records[-1].dcop

    # Shape: [col_num, row_num]
    v_node = dcop.v_sl_node
    # Shape: [col_num]
    v_drive = dcop.v_sl_drive
    assert float((v_drive.unsqueeze(-1) - v_node).abs().min()) > 0.0, (
        "witness must draw current, else node and boundary coincide"
    )

    slope__fJ = float(raised) - float(base)
    at_node = _V_DD_BL__V * delta__fF * float(v_node.abs().sum())
    at_drive = _V_DD_BL__V * delta__fF * _ROW_NUM * float(v_drive.abs().sum())

    assert slope__fJ == pytest.approx(at_node, rel=1e-10)
    assert slope__fJ != pytest.approx(at_drive, rel=1e-10)
