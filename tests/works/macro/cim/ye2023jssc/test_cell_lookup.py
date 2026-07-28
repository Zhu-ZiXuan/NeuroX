"""CPU-only eager tests for the ye2023jssc WH-2T1R lookup cell.

Laws (config = arbitrary hand-written witness, not the assertion target):

  * ``lookup_i_t2`` selects the programmed-state IN=1 current when the input
    bit is high and the state-independent IN=0 floor when it is low; a
    LRS-programmed cell sources more IN=1 current than a HRS one, and the
    IN=0 floor is the same across states,
  * step1 is delegated verbatim to the linear base — ``solve_dc`` returns an
    :class:`XbarCell1t1rDcop` whose ``i__uA == g_cell * (v_bl - v_sl)`` and
    ``v_x__V == v_bl - vx_ratio * (v_bl - v_sl)`` exactly,
  * ``compute_dynamic_energy`` is the PER-ACCESS pair only: the WL gate load on
    every cell plus the selected row's closed-form X dip-recharge
    ``c_x * v_bl^2 * vx_ratio_on``. It ignores the SL node and the BL node
    capacitance entirely (both are the macro's per-vector concern), so it is
    invariant to ``c_sl__fF`` / ``c_bl__fF`` and collapses to the WL term where
    the WL is off or the column is driven input-low,
  * the ``(config, policy)`` pair dispatches through the ``XbarCell1t1r``
    registry to :class:`Ye2023Jssc2t1rCell`.

Runs eagerly (dynamo disabled) so nothing is unrolled; tiny CPU shapes.
"""

from __future__ import annotations

import dataclasses
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
_FLOOR__uA = 0.02  # IN=0 off-cell floor, state-independent
_LEAK_IN1__uA = 0.03  # (HRS, IN=1) weight leakage
_UNIT_IN1__uA = 0.5  # (LRS, IN=1) unit-scale on-current I_unit

_C_WL__fF = 1.5
_C_X__fF = 0.7
_C_BL__fF = 2.0
_C_SL__fF = 3.0
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
        c_bl__fF=_C_BL__fF,
        c_x__fF=_C_X__fF,
        c_sl__fF=_C_SL__fF,
        c_wl__fF=_C_WL__fF,
        g_cell_off_table__uS=(1.0, 5.0),
        g_cell_on_table__uS=(2.0, 10.0),
        vx_ratio_off_table=(0.3, 0.4),
        vx_ratio_on_table=_VX_RATIO_ON,
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
        # i_t2_table__uA[input_bit][state]; state axis = (HRS, LRS).
        i_t2_table__uA=(
            (_FLOOR__uA, _FLOOR__uA),  # IN=0: state-independent floor
            (_LEAK_IN1__uA, _UNIT_IN1__uA),  # IN=1: HRS leak vs LRS unit current
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


def _snapshot(cell: Ye2023Jssc2t1rCell, v_wl: Tensor) -> Ye2023Jssc2t1rCellSnap:
    return cell.snapshot(
        control=v_wl,
        shape=tuple(v_wl.shape),
        multi_coords=None,
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


def test_registry_dispatch() -> None:
    """The (config, policy) pair selects the WH-2T1R lookup cell."""
    cell = _build_cell(_build_config(), (2, 3))
    assert isinstance(cell, Ye2023Jssc2t1rCell)
    assert cell.w_state_num == 2


def test_lookup_i_t2_selects_state_and_input() -> None:
    """IN=1 returns the programmed-state current; IN=0 returns the floor."""
    inst_shape = (2, 3)
    cell = _build_cell(_build_config(), inst_shape)
    # A mixed HRS/LRS state pattern over (col, row).
    state = torch.tensor([[_HRS, _LRS, _HRS], [_LRS, _HRS, _LRS]])
    cell.program(state)
    snap = _snapshot(cell, torch.full(inst_shape, _V_WL_SEL__V, dtype=_DTYPE))

    lrs = state == _LRS
    hrs = state == _HRS

    # IN=1 -> the programmed-state IN=1 lookup (unit scale, m = 1).
    i_high = cell.lookup_i_t2(torch.ones(inst_shape, dtype=torch.bool), snap)
    assert torch.equal(i_high, snap.i_t2_in1__uA)
    # LRS cell delivers strictly more IN=1 current than HRS.
    assert i_high[lrs].min() > i_high[hrs].max()

    # IN=0 -> the off-cell floor, identical across states.
    i_low = cell.lookup_i_t2(torch.zeros(inst_shape, dtype=torch.bool), snap)
    assert torch.equal(i_low, snap.i_t2_in0__uA)
    assert i_low.min() == i_low.max()  # floor is state-independent

    # Per-cell mixed input: high where LRS, low where HRS.
    i_mixed = cell.lookup_i_t2(lrs, snap)
    assert torch.equal(i_mixed[lrs], snap.i_t2_in1__uA[lrs])
    assert torch.equal(i_mixed[hrs], snap.i_t2_in0__uA[hrs])


def test_step1_delegation_unchanged() -> None:
    """solve_dc matches the linear divider formulas exactly."""
    inst_shape = (2, 3)
    config = _build_config()
    state = torch.tensor([[_HRS, _LRS, _HRS], [_LRS, _HRS, _LRS]])
    cell = _programmed_cell(config, inst_shape, state)

    v_wl = torch.full(inst_shape, _V_WL_SEL__V, dtype=_DTYPE)  # above threshold -> on params
    snap = _snapshot(cell, v_wl)
    v_bl = torch.full(inst_shape, _V_BL_IN1__V, dtype=_DTYPE)
    v_sl = torch.zeros(inst_shape, dtype=_DTYPE)

    dcop = cell.solve_dc(v_bl, v_sl, snap)
    assert isinstance(dcop, XbarCell1t1rDcop)

    on = v_wl > config.v_wl_on_threshold__V
    g_cell = torch.where(on, snap.g_cell_on__uS, snap.g_cell_off__uS)
    vx_ratio = torch.where(on, snap.vx_ratio_on, snap.vx_ratio_off)
    dv = v_bl - v_sl
    assert torch.equal(dcop.i__uA, g_cell * dv)
    assert torch.equal(dcop.v_x__V, v_bl - vx_ratio * dv)
    assert torch.equal(dcop.di_dvbl__uS, g_cell)
    assert torch.equal(dcop.di_dvsl__uS, -g_cell)


# ---------------------------------------------------------------------------
# Per-access capacitance: WL gate + selected-row X dip-recharge only
# ---------------------------------------------------------------------------


def _cell_energy(config: Ye2023Jssc2t1rCellConfig, *, v_wl: Tensor, v_bl: Tensor, state: Tensor) -> Tensor:
    inst_shape = tuple(state.shape)
    cell = _programmed_cell(config, inst_shape, state)
    snap = _snapshot(cell, v_wl.expand(inst_shape))
    v_sl = torch.zeros(inst_shape, dtype=_DTYPE)
    dcop = cell.solve_dc(v_bl, v_sl, snap)
    return cell.compute_dynamic_energy(v_bl, v_sl, dcop, snap)


def test_per_access_energy_is_wl_gate_plus_x_dip() -> None:
    """Per-cell per-access energy == ``c_wl*V_WL^2 + c_x*V_BL^2*vx_ratio_on`` (WL on only)."""
    config = _build_config()
    state = torch.tensor([[_HRS, _LRS], [_LRS, _HRS]])
    inst_shape = tuple(state.shape)
    # Row 0 selected (WL high), row 1 unselected; column 0 input-high, column 1 low.
    v_wl = torch.tensor([[_V_WL_SEL__V, 0.0]], dtype=_DTYPE).expand(inst_shape)
    v_bl = torch.tensor([[_V_BL_IN1__V], [0.0]], dtype=_DTYPE).expand(inst_shape)

    got = _cell_energy(config, v_wl=v_wl, v_bl=v_bl, state=state)

    vx_ratio_on = torch.tensor(_VX_RATIO_ON, dtype=_DTYPE)[state]
    wl_on = v_wl > _V_WL_ON_THRESHOLD__V
    expected = _C_WL__fF * v_wl.square() + _C_X__fF * v_bl.square() * vx_ratio_on * wl_on
    assert torch.equal(got, expected)

    # Unselected row: the WL gate term alone (no dip -> the X node never moves).
    assert torch.equal(got[:, 1], _C_WL__fF * v_wl[:, 1].square())
    # Input-low column of the SELECTED row: also the WL term alone (V_BL = 0).
    assert float(got[1, 0]) == pytest.approx(_C_WL__fF * _V_WL_SEL__V**2)
    # Input-high column of the selected row carries the dip on top.
    assert float(got[0, 0]) > float(got[1, 0])


def test_per_access_energy_ignores_sl_and_bl_node_caps() -> None:
    """The per-access term reads neither ``c_sl__fF`` nor ``c_bl__fF``."""
    base = _build_config()
    state = torch.tensor([[_HRS, _LRS], [_LRS, _HRS]])
    inst_shape = tuple(state.shape)
    v_wl = torch.full((1, inst_shape[1]), _V_WL_SEL__V, dtype=_DTYPE).expand(inst_shape)
    v_bl = torch.full(inst_shape, _V_BL_IN1__V, dtype=_DTYPE)

    got = _cell_energy(base, v_wl=v_wl, v_bl=v_bl, state=state)
    variants = {
        "c_sl__fF": dataclasses.replace(base, c_sl__fF=10.0 * _C_SL__fF),
        "c_bl__fF": dataclasses.replace(base, c_bl__fF=10.0 * _C_BL__fF),
    }
    for field, config in variants.items():
        moved = _cell_energy(config, v_wl=v_wl, v_bl=v_bl, state=state)
        assert torch.equal(moved, got), f"per-access energy moved with {field}"
