"""Laws for the scheme-local serial-column 1T1R array.

Hand-built tiny geometry, eager, CPU-friendly. Three laws:

  * OFF-COLUMN EQUIVALENCE (the off-column definition, executed): a padded
    full-width solve — every physical column materialized, the columns outside
    the active slot grounded (BL clamp ``v_ref = 0``, ``r_out = 0``, SL at 0,
    WL held by the plane) — equals the active-only serial-column solve
    elementwise on the active columns, and every off column sits at
    identically zero node voltages and current (hence zero BL/SL/node cap
    energy). Off columns therefore never need materializing.
  * TRUE-SHAPE LAW: the macro fabricates its cablc / sl_driver clamp banks
    at ``(gn, 2, w_digit)`` and its dswct / sinwp_sc / pn_isub readout
    modules at ``(gn, 2)`` / ``(gn, 2)`` / ``(gn,)``, all derived from the
    config (no magic numbers) and prefix-safe under a fabrication prefix.
  * WL-PER-PLANE LAW: the WL cap energy is billed once per plane, so the
    total array cap energy is invariant to how the same physical columns are
    factored into (serial, lane) slots — a per-slot WL billing would scale
    the WL wire term with ``serial``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog import (
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig, SolverProber
from neurox.works.macro.cim.xue2020jssc import SerialColumnXbarArray

from ._utils import build_config, build_macro

_DTYPE = torch.float64
_ROW_NUM = 4
_V_BLC__V = 0.3
_V_WL_ON__V = 0.9


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Hand-built tiny witness pieces
# ---------------------------------------------------------------------------


def _array_config(*, wl_first_c__fF: float = 0.4, wl_segment_c__fF: float = 0.1) -> XbarArray1t1rConfig:
    """Near-ideal linear-cell array config with distinct WL first/segment caps.

    ``wl_first_c != wl_segment_c`` on purpose: a per-slot WL wire billing then
    produces a serial-dependent total, which the WL-per-plane law catches.
    """
    return XbarArray1t1rConfig(
        row_first_space__um=1.0,
        row_cell_space__um=1.0,
        col_first_space__um=1.0,
        col_cell_space__um=1.0,
        bl_first_r__MOhm=2.0e-5,
        bl_first_c__fF=0.1,
        bl_segment_r__MOhm=5.0e-6,
        bl_segment_c__fF=0.1,
        sl_first_r__MOhm=2.0e-5,
        sl_first_c__fF=0.1,
        sl_segment_r__MOhm=5.0e-6,
        sl_segment_c__fF=0.1,
        wl_first_r__MOhm=2.0e-5,
        wl_first_c__fF=wl_first_c__fF,
        wl_segment_r__MOhm=5.0e-6,
        wl_segment_c__fF=wl_segment_c__fF,
        cell_config=XbarCell1t1rLinearConfig(
            c_bl__fF=0.2,
            c_x__fF=0.3,
            c_sl__fF=0.1,
            c_wl__fF=0.2,
            g_cell_on_table__uS=(0.0, 100.0),
            g_cell_off_table__uS=(0.0, 0.0),
            vx_ratio_on_table=(0.5, 0.5),
            vx_ratio_off_table=(0.5, 0.5),
            v_wl_on_threshold__V=0.5,
        ),
        solver_config=NestedParallelRailSolverConfig(n_outer=3, n_inner=3),
        latency_per_op__ns=0.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=1.0,
    )


def _array_policy() -> XbarArray1t1rPolicy:
    return XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=0)


def _voltage_driver(inst_shape: tuple[int, ...]) -> VoltageDriver:
    """Ideal (``r_out = 0``) all-off clamp driver; the reference is injected per solve."""
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
        inst_shape=inst_shape,
        dtype=_DTYPE,
        T__K=300.0,
    )
    driver.eval()
    driver.fabricate()
    return driver


def _serial_array(slot_map: Tensor, *, config: XbarArray1t1rConfig | None = None) -> SerialColumnXbarArray:
    array = SerialColumnXbarArray(
        config=config if config is not None else _array_config(),
        policy=_array_policy(),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=slot_map.numel(),
        slot_map=slot_map,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    return array


def _physical_states(col_num: int) -> Tensor:
    """Deterministic mixed 0/1 physical state grid."""
    gen = torch.Generator().manual_seed(7)
    return torch.randint(0, 2, (col_num, _ROW_NUM), generator=gen, dtype=torch.long)


def _plane() -> Tensor:
    """One WL plane with mixed on/off rows [V]."""
    return torch.tensor([_V_WL_ON__V, 0.0, _V_WL_ON__V, _V_WL_ON__V], dtype=_DTYPE)


# ---------------------------------------------------------------------------
# Off-column equivalence law (S9.1)
# ---------------------------------------------------------------------------


def test_off_column_equivalence_law() -> None:
    """Padded full-width solve == active-only solve elementwise; off columns identically zero."""
    # Shape: [serial, gn, polarity, w_digit]
    slot_map = torch.arange(4).reshape(2, 1, 2, 1)
    serial, act = 2, 2
    states = _physical_states(4)
    plane = _plane()

    # --- Active-only: the serial-column array (off columns never materialized) ---

    array = _serial_array(slot_map)
    array.program(states)
    bl_driver = _voltage_driver((1, 2, 1))
    sl_driver = _voltage_driver((1, 2, 1))
    with SolverProber() as active_probe, torch.no_grad():
        steady = array.solve_array(
            plane,
            bl_driver=bl_driver,
            bl_v_ref__V=torch.tensor(_V_BLC__V, dtype=_DTYPE),
            sl_driver=sl_driver,
            sl_v_ref__V=torch.zeros((), dtype=_DTYPE),
        )
    # Shape: [serial, gn, 2, wd] -> [serial, act]
    i_active = steady.i_bl_port__uA.flatten(-3)
    active_dcop = active_probe.records[-1].dcop

    # --- Padded: full-width kernel solve, off columns grounded per the definition ---

    kernel_array = XbarArray1t1r(
        config=_array_config(),
        policy=_array_policy(),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=4,
        dtype=_DTYPE,
        T__K=300.0,
    )
    kernel_array.eval()
    kernel_array.fabricate()
    kernel_array.program(states)

    # Per-(slot, column) references, injected straight into the kernel solve:
    # the slot's own columns clamp at V_BLC / 0 (SL); every other column is
    # grounded (0 V both rails). The injected reference broadcasts onto the
    # per-call solve shape, so the off-column definition needs no stub driver.
    active_mask = torch.zeros((serial, 4), dtype=torch.bool)
    for s in range(serial):
        active_mask[s, slot_map[s].reshape(-1)] = True
    bl_refs = torch.where(active_mask, torch.tensor(_V_BLC__V, dtype=_DTYPE), torch.tensor(0.0, dtype=_DTYPE))
    sl_refs = torch.zeros((serial, 4), dtype=_DTYPE)

    with SolverProber() as padded_probe, torch.no_grad():
        padded = kernel_array.solve_array(
            plane.expand(serial, _ROW_NUM),  # the same plane, held across both slots
            bl_driver=_voltage_driver((4,)),
            bl_v_ref__V=bl_refs,
            sl_driver=_voltage_driver((4,)),
            sl_v_ref__V=sl_refs,
        )
    padded_dcop = padded_probe.records[-1].dcop

    # --- Active columns: padded == active-only, elementwise ---

    for s in range(serial):
        cols = slot_map[s].reshape(-1)
        torch.testing.assert_close(padded.i_bl_port__uA[s, cols], i_active[s], rtol=0.0, atol=0.0)
        torch.testing.assert_close(padded_dcop.v_bl_node[s, cols], active_dcop.v_bl_node[s, :act], rtol=0.0, atol=0.0)
        torch.testing.assert_close(padded_dcop.v_sl_node[s, cols], active_dcop.v_sl_node[s, :act], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            padded_dcop.cell.v_x__V[s, cols], active_dcop.cell.v_x__V[s, :act], rtol=0.0, atol=0.0
        )

    # --- Off columns: identically zero (voltages, current => zero cap energy) ---

    off = ~active_mask
    assert (padded.i_bl_port__uA[off] == 0.0).all(), "off-column BL port current must be identically 0"
    assert (padded_dcop.v_bl_node[off] == 0.0).all(), "off-column BL nodes must sit at 0 V"
    assert (padded_dcop.v_sl_node[off] == 0.0).all(), "off-column SL nodes must sit at 0 V"
    assert (padded_dcop.cell.v_x__V[off] == 0.0).all(), "off-column access nodes must sit at 0 V"
    assert (padded_dcop.cell.i__uA[off] == 0.0).all(), "off-column cell currents must be identically 0"


# ---------------------------------------------------------------------------
# True-shape law (S9.3)
# ---------------------------------------------------------------------------


def test_true_shape_law(device: torch.device) -> None:
    """Clamp banks + readout modules fabricate at the true hardware counts derived from the config."""
    for w_digit_num, mux_factor in ((2, 2), (3, 2)):
        config = build_config(w_digit_num=w_digit_num, mux_factor=mux_factor)
        for inst in ((), (2,)):
            macro = build_macro(config, device=device, inst_shape=inst)
            gn = macro.col_num // config.mux_factor
            lane = (gn, 2, config.w_digit_num)
            for name, trailing in (
                ("cablc", lane),
                ("sl_driver", lane),
                ("dswct", (gn, 2)),
                ("sinwp_sc", (gn, 2)),
                ("pn_isub", (gn,)),
            ):
                module = getattr(macro, name)
                want = (*inst, *trailing)
                assert module.inst_shape == want, f"{name}.inst_shape {module.inst_shape} != {want}"
                assert module.inst_count == math.prod(inst) * math.prod(trailing)
            # The array seats every physical column: col_num = serial * gn * 2 * wd.
            assert macro.array.weight_grid_shape == (
                *inst,
                config.mux_factor * gn * 2 * config.w_digit_num,
                macro.row_num,
            )


def test_slot_map_must_be_bijection() -> None:
    """A slot map with a repeated physical column is rejected at init."""
    bad = torch.arange(4).reshape(2, 1, 2, 1).clone()
    bad[1, 0, 1, 0] = 0  # duplicates column 0, drops column 3
    with pytest.raises(ValueError, match="bijection"):
        _serial_array(bad)


# ---------------------------------------------------------------------------
# WL-per-plane law (S9.4)
# ---------------------------------------------------------------------------


def test_wl_cap_energy_independent_of_serial() -> None:
    """The array cap energy is invariant to the serial factoring of the same physical columns.

    Both factorings solve the identical physical columns under identical
    clamps, so the BL/SL wire and cell node terms match column-for-column;
    the only serial-sensitive term would be a per-slot WL billing, which the
    distinct ``wl_first_c != wl_segment_c`` witness would expose.
    """
    states = _physical_states(4)
    plane = _plane()

    def run(slot_map: Tensor, lane_shape: tuple[int, ...]) -> tuple[float, Tensor]:
        array = _serial_array(slot_map)
        array.program(states)
        bl_driver = _voltage_driver(lane_shape)
        sl_driver = _voltage_driver(lane_shape)
        with NeuroxProfiler() as prof, torch.no_grad():
            steady = array.solve_array(
                plane,
                bl_driver=bl_driver,
                bl_v_ref__V=torch.tensor(_V_BLC__V, dtype=_DTYPE),
                sl_driver=sl_driver,
                sl_v_ref__V=torch.zeros((), dtype=_DTYPE),
            )
        report = prof.report(array)
        # Physical-order currents: seats flatten back to columns 0..3 for both maps.
        return report.total_dynamic_energy__fJ, steady.i_bl_port__uA.reshape(-1)

    # The same 4 physical columns: one slot of 4 lanes vs two slots of 2 lanes.
    e_serial1, i_serial1 = run(torch.arange(4).reshape(1, 1, 2, 2), (1, 2, 2))
    e_serial2, i_serial2 = run(torch.arange(4).reshape(2, 1, 2, 1), (1, 2, 1))

    assert e_serial1 > 0.0
    assert e_serial1 == pytest.approx(e_serial2, rel=1e-12), (
        f"array cap energy moved with the serial factoring: {e_serial1} vs {e_serial2} "
        "(WL caps must be billed once per plane, independent of serial)"
    )
    # Placement law: both factorings solve the same physical columns to the same DC point.
    torch.testing.assert_close(i_serial1, i_serial2, rtol=0.0, atol=0.0)
