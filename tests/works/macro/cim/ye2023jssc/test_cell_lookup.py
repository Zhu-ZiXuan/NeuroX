"""CPU-only eager tests for the ye2023jssc WH-2T1R lookup cell.

Laws (config = arbitrary hand-written witness, not the assertion target):

  * `i_t2__uA` branches on TWO independent facts — is this cell's row pair
    driven (its own WL against the configured threshold), and which calibration
    operating point its solved `V_X` sits at (`V_X = 0` floor, `V_X > 0`
    drive);
  * a WL at or below the threshold sources EXACTLY zero whatever the branch
    solved to, `V_X > 0` included — the unselected row is pinned, so its T2
    drain is undriven;
  * a driven cell classifies on the operating point alone: `V_X = 0` reads the
    floor entry of the programmed state, `V_X > 0` the drive entry, and an LRS
    cell drives more than an HRS one;
  * step1 is delegated verbatim to the linear base — `solve_dc` returns an
    `XbarCell1t1rDcop` whose `i__uA == g_cell__uS * (v_bl__V - v_sl__V)` and
    `v_x__V == v_bl__V - vx_ratio * (v_bl__V - v_sl__V)` exactly,
  * the cell adds NO energy model of its own: the capacitive law is the kernel
    1T1R cell's supply-draw one, pinned where it lives,
  * the `(config, policy)` pair dispatches through the `XbarCell1t1r`
    registry to `Ye2023Jssc2t1rCell`.

Runs eagerly (dynamo disabled) so nothing is unrolled; tiny CPU shapes.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.xbar.cell import XbarCell1t1r, XbarCell1t1rDcop
from neurox.works.macro.cim.ye2023jssc.cell import (
    Ye2023Jssc2t1rCell,
    Ye2023Jssc2t1rCellConfig,
    Ye2023Jssc2t1rCellPolicy,
    Ye2023Jssc2t1rCellSnap,
)

# WH-2T1R has two weight states: HRS = 0, LRS = 1.
_HRS = 0
_LRS = 1

# Witness table values (arbitrary; the tests assert laws, not these numbers).
_FLOOR__uA = 0.02  # V_X = 0 off-cell floor, state-independent
_LEAK_DRIVE__uA = 0.03  # (HRS, drive point) weight leakage
_UNIT_DRIVE__uA = 0.5  # (LRS, drive point) unit-scale on-current I_unit

_VX_RATIO_ON = (0.5, 0.6)
_V_WL_ON_THRESHOLD__V = 0.3
_V_WL_SEL__V = 0.6
_V_BL_IN1__V = 0.3

_DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — do not unroll any compiled leaf."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build_config() -> Ye2023Jssc2t1rCellConfig:
    """A tiny two-state (HRS/LRS) witness config."""
    return Ye2023Jssc2t1rCellConfig(
        g_cell_off_table__uS=(1.0, 5.0),
        g_cell_on_table__uS=(2.0, 10.0),
        vx_ratio_off_table=(0.3, 0.4),
        vx_ratio_on_table=_VX_RATIO_ON,
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
        # i_t2_table__uA[operating_point][state]; state axis = (HRS, LRS).
        i_t2_table__uA=(
            (_FLOOR__uA, _FLOOR__uA),  # V_X = 0: state-independent floor
            (_LEAK_DRIVE__uA, _UNIT_DRIVE__uA),  # V_X > 0: HRS leak vs LRS unit current
        ),
    )


def _build_cell(config: Ye2023Jssc2t1rCellConfig, inst_shape: tuple[int, ...]) -> Ye2023Jssc2t1rCell:
    cell = XbarCell1t1r.from_config(
        config=config,
        policy=Ye2023Jssc2t1rCellPolicy(),
        inst_shape=inst_shape,
        dtype=_DTYPE,
        T__K=300.0,
    )
    assert isinstance(cell, Ye2023Jssc2t1rCell)
    return cell


def _snapshot(cell: Ye2023Jssc2t1rCell, v_wl__V: Tensor) -> Ye2023Jssc2t1rCellSnap:
    """Snapshot at a per-row WL drive expanded onto the cell grid, as an array does."""
    return cell.snapshot(
        control=v_wl__V.unsqueeze(-2).expand(cell.inst_shape),
        shape=cell.inst_shape,
        t_elapsed=0.0,
    )


def _programmed_cell(
    config: Ye2023Jssc2t1rCellConfig,
    inst_shape: tuple[int, ...],
    state: Tensor,
) -> Ye2023Jssc2t1rCell:
    cell = _build_cell(config, inst_shape)
    cell.program(state)
    return cell


def _solved(
    cell: Ye2023Jssc2t1rCell,
    snap: Ye2023Jssc2t1rCellSnap,
    *,
    v_bl__V: float,
    inst_shape: tuple[int, ...],
) -> XbarCell1t1rDcop:
    """Settle the branch at a hand-written drive level against a grounded SL."""
    v_bl__V = torch.full(inst_shape, v_bl__V, dtype=_DTYPE)
    v_sl__V = torch.zeros(inst_shape, dtype=_DTYPE)
    return cell.solve_dc(v_bl__V, v_sl__V, snap)


def test_registry_dispatch() -> None:
    """The (config, policy) pair selects the WH-2T1R lookup cell."""
    cell = _build_cell(_build_config(), (2, 3))
    assert isinstance(cell, Ye2023Jssc2t1rCell)
    assert cell.w_state_num == 2


def test_i_t2_selects_state_and_operating_point() -> None:
    """A driven cell reads the drive entry above V_X = 0 and the floor entry at it."""
    inst_shape = (2, 3)
    cell = _build_cell(_build_config(), inst_shape)
    # A mixed HRS/LRS state pattern over (col, row).
    state = torch.tensor([[_HRS, _LRS, _HRS], [_LRS, _HRS, _LRS]])
    cell.program(state)
    snap = _snapshot(cell, torch.full((inst_shape[1],), _V_WL_SEL__V, dtype=_DTYPE))

    lrs = state == _LRS
    hrs = state == _HRS

    # A driven BL puts every branch at V_X > 0 -> the programmed state's drive entry.
    driven = _solved(cell, snap, v_bl__V=_V_BL_IN1__V, inst_shape=inst_shape)
    assert bool((driven.v_x__V > 0.0).all())
    i_drive = cell.i_t2__uA(driven, snap)
    assert torch.equal(i_drive, snap.i_t2_drive__uA)
    # An LRS cell delivers strictly more drive current than an HRS one.
    assert i_drive[lrs].min() > i_drive[hrs].max()

    # Both boundaries grounded -> V_X = 0 -> the floor, identical across states.
    floored = _solved(cell, snap, v_bl__V=0.0, inst_shape=inst_shape)
    assert torch.equal(floored.v_x__V, torch.zeros_like(floored.v_x__V))
    i_floor = cell.i_t2__uA(floored, snap)
    assert torch.equal(i_floor, snap.i_t2_floor__uA)
    assert i_floor.min() == i_floor.max()  # the floor is state-independent

    # Per-cell mixed drive: the classification is per cell, not per call.
    v_bl__V = torch.where(lrs, torch.tensor(_V_BL_IN1__V, dtype=_DTYPE), torch.zeros((), dtype=_DTYPE))
    mixed = cell.solve_dc(v_bl__V, torch.zeros(inst_shape, dtype=_DTYPE), snap)
    i_mixed = cell.i_t2__uA(mixed, snap)
    assert torch.equal(i_mixed[lrs], snap.i_t2_drive__uA[lrs])
    assert torch.equal(i_mixed[hrs], snap.i_t2_floor__uA[hrs])


def test_undriven_row_sources_exactly_zero() -> None:
    """A WL at or below the threshold contributes nothing, V_X > 0 included.

    An unselected row has WL and TBL both held at ground, so its cells' T2
    drains are undriven whatever the divider settled to on the BL side.
    """
    inst_shape = (2, 3)
    cell = _programmed_cell(_build_config(), inst_shape, torch.tensor([[_HRS, _LRS, _HRS], [_LRS, _HRS, _LRS]]))

    for v_wl__V in (0.0, _V_WL_ON_THRESHOLD__V):  # off, and exactly AT the threshold
        snap = _snapshot(cell, torch.full((inst_shape[1],), v_wl__V, dtype=_DTYPE))
        for v_bl__V in (0.0, _V_BL_IN1__V):
            dcop = _solved(cell, snap, v_bl__V=v_bl__V, inst_shape=inst_shape)
            i_t2 = cell.i_t2__uA(dcop, snap)
            assert torch.equal(i_t2, torch.zeros_like(i_t2)), (v_wl__V, v_bl__V)
    # The pinned case is not vacuous: a driven BL does leave V_X above zero.
    snap_off = _snapshot(cell, torch.zeros(inst_shape[1], dtype=_DTYPE))
    pinned = _solved(cell, snap_off, v_bl__V=_V_BL_IN1__V, inst_shape=inst_shape)
    assert bool((pinned.v_x__V > 0.0).all())


def test_i_t2_keeps_the_snap_dtype() -> None:
    """The zero branch is a weak scalar: the result keeps the cell's dtype."""
    inst_shape = (2, 3)
    cell = _programmed_cell(_build_config(), inst_shape, torch.zeros(inst_shape, dtype=torch.long))
    snap = _snapshot(cell, torch.full((inst_shape[1],), _V_WL_SEL__V, dtype=_DTYPE))
    dcop = _solved(cell, snap, v_bl__V=_V_BL_IN1__V, inst_shape=inst_shape)
    assert cell.i_t2__uA(dcop, snap).dtype == _DTYPE


def test_step1_delegation_unchanged() -> None:
    """solve_dc matches the linear divider formulas exactly."""
    inst_shape = (2, 3)
    config = _build_config()
    state = torch.tensor([[_HRS, _LRS, _HRS], [_LRS, _HRS, _LRS]])
    cell = _programmed_cell(config, inst_shape, state)

    v_wl__V = torch.full((inst_shape[1],), _V_WL_SEL__V, dtype=_DTYPE)  # above threshold -> on params
    snap = _snapshot(cell, v_wl__V)
    v_bl__V = torch.full(inst_shape, _V_BL_IN1__V, dtype=_DTYPE)
    v_sl__V = torch.zeros(inst_shape, dtype=_DTYPE)

    dcop = cell.solve_dc(v_bl__V, v_sl__V, snap)
    assert isinstance(dcop, XbarCell1t1rDcop)

    on = v_wl__V > config.v_wl_on_threshold__V
    g_cell__uS = torch.where(on, snap.g_cell_on__uS, snap.g_cell_off__uS)
    vx_ratio = torch.where(on, snap.vx_ratio_on, snap.vx_ratio_off)
    dv__V = v_bl__V - v_sl__V
    assert torch.equal(dcop.i__uA, g_cell__uS * dv__V)
    assert torch.equal(dcop.v_x__V, v_bl__V - vx_ratio * dv__V)
    assert torch.equal(dcop.di_dvbl__uS, g_cell__uS)
    assert torch.equal(dcop.di_dvsl__uS, -g_cell__uS)
