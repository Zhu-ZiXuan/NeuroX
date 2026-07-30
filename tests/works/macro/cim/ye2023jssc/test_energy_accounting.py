"""Eager energy-accounting laws for the ye2023jssc WH-2T1R CIM macro.

Pins the Fig.19 branch-ownership and channel-billing contract on the hand-built
analytic witness (``_utils.build_config``). Every check is a LAW read off the
profiler report: the coefficients are whatever the deterministic all-off analog
chain produces, and the assertions constrain how the billed energy MOVES.

Coverage:

  * the Fig.19 blocks appear under their exact channel / module names — the
    per-access array caps row ``array``, the macro-owned ``.bl_cond`` /
    ``.dl_cond`` conduction channels, the macro-owned per-vector ``.bl_cap``
    charge channel, the RS-CSA row ``rscsa``, and the two flat peripheral
    channels ``.mux_driver`` / ``.timing_ctrl`` — all positive, each billed by
    the RIGHT module (array / RS-CSA self-bill un-channelled; the five channels
    are the macro root's own events; no ``cell`` self-bill row leaks through),
    and they ADD UP to the report's total dynamic energy,
  * branch ownership: both conduction channels reconcile EXACTLY against an
    independent re-solve oracle over ONE window — ``.bl_cond == V_BL_in1 *
    sum(I_BL_port) * T_AC`` and ``.dl_cond == V_DD_core * sum(I_TBL_raw) * T_AC``
    with the RAW row current (leakage floor included, before PH0),
  * the window is derived and shared: stretching the RS-CSA phase set stretches
    both conduction channels by the same factor and leaves every capacitive row
    untouched,
  * the window is the EXECUTED one: lowering ``adc_bits`` drops compare phases,
    so both conduction channels, the latency, and the readout's zero-residue
    baseline scale by the executed-window ratio while the capacitive rows and
    the per-op control lumps do not move; at ``adc_bits == adc_max_bits`` every
    value is the nominal one,
  * caps are split three ways: the array bills only per-access WL-side terms
    (invariant to the SL and BL capacitances), while the macro's per-vector
    ``.bl_cap`` is the BL-column charge ``V_BL^2 * C_col`` per input-high
    physical column — linear in the active-input count, zero at zero input, and
    the only row that moves with ``c_bl__fF`` / the BL wire caps,
  * the all-off floor: with zero input the DL branch bills exactly the derived
    PH0 current on every output, the BL channels vanish, and the array collapses
    to its closed-form WL-only value,
  * the RS-CSA is E_fixed-dominant: when ``e_fixed`` dwarfs the per-code SAR
    energy the per-conversion energy is code-independent to within a few percent,
  * latency is the macro's SOLE emission: exactly one event of
    ``T_AC * serial_rounds``, scaling with the serial round count,
  * static leakage reconciles: ``leakage_energy == leakage_power * total_latency``
    with ``leakage_power`` the sum over the ``collect_static`` seats.

Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is not
unrolled.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler, ProfilerReport
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig

from ._utils import (
    DTYPE,
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    V_BL_IN1__V,
    V_WL_SEL__V,
    W_MAX,
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    build_config,
    build_macro,
    cell_config,
    expected_ph0__uA,
)

# The Fig.19 blocks as they surface in ``energy_by_name`` (macro root == "").
_ARRAY_CAPS = "array"
_RSCSA = "rscsa"
_MACRO_CHANNELS = ("bl_cond", "dl_cond", "bl_cap", "mux_driver", "timing_ctrl")
_FIG19_KEYS = (_ARRAY_CAPS, ".bl_cond", ".dl_cond", ".bl_cap", _RSCSA, ".mux_driver", ".timing_ctrl")


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    config: Ye2023JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    adc_bits: int = TINY_ADC_BITS,
) -> tuple[Ye2023JsscCimMacro, NeuroxProfiler, ProfilerReport]:
    """Build + fabricate a fresh macro, program ``w``, profile one VMM on ``x``."""
    macro = build_macro(config, input_num=w.shape[-2], output_num=w.shape[-1], device=device)
    macro.program(w.to(device))
    with NeuroxProfiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=adc_bits)
    return macro, prof, prof.report(macro)


def _drive(macro: Ye2023JsscCimMacro, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Rebuild the exact drive ``vec_mat_mul`` constructs, from the config alone.

    Returns the one-hot-per-output word lines, the per-column BL reference
    voltages broadcast over the output leading, and the plane-major input
    indicator (weight planes tiled, redundant planes forced input-0).
    """
    cfg = macro.config
    device = x.device
    n_weight_plane = cfg.w_digit_num
    n_redundant_plane = len(cfg.array_config.redundant_radix)
    input_num = macro.row_num
    output_num = macro.col_num

    x_long = x.long()
    leading = x_long.shape[:-1]
    x_weight = (
        x_long.unsqueeze(-2).expand(*leading, n_weight_plane, input_num).reshape(*leading, n_weight_plane * input_num)
    )
    x_tiled = torch.cat((x_weight, x_weight.new_zeros((*leading, n_redundant_plane * input_num))), dim=-1)
    v_bl = torch.where(
        x_tiled > 0,
        torch.tensor(cfg.v_bl_in1__V, dtype=DTYPE, device=device),
        torch.tensor(0.0, dtype=DTYPE, device=device),
    )
    wl_onehot = cfg.v_wl_sel__V * torch.eye(output_num, dtype=DTYPE, device=device)
    v_wl = wl_onehot.expand(*leading, output_num, output_num)
    bl_v_ref = v_bl.unsqueeze(-2).expand(*leading, output_num, x_tiled.shape[-1])
    return v_wl, bl_v_ref, x_tiled


def _conduction_oracle(macro: Ye2023JsscCimMacro, x: Tensor) -> tuple[float, float]:
    """Independent ``(.bl_cond, .dl_cond)`` [fJ] from a re-solve over the derived window.

    Re-runs the array DC solve OUTSIDE any profiler (so it logs nothing of its
    own) and applies the ONE access window every conduction branch rides::

        bl_cond = V_BL_in1  * sum(I_BL_port) * T_AC
        dl_cond = V_DD_core * sum(I_TBL_raw) * T_AC
    """
    cfg = macro.config
    v_wl, bl_v_ref, _x_tiled = _drive(macro, x)
    steady = macro.array.solve(
        v_wl,
        bl_driver=macro.bl_driver,
        bl_v_ref__V=bl_v_ref,
        sl_driver=macro.sl_driver,
        sl_v_ref__V=torch.tensor(cfg.v_sl__V, dtype=DTYPE, device=x.device),
    )
    t_ac__ns = float(macro.t_ac__ns)
    bl_cond = float((cfg.v_bl_in1__V * steady.i_bl_port__uA).sum() * t_ac__ns)
    dl_cond = float((cfg.v_dd_core__V * steady.i_tbl__uA).sum() * t_ac__ns)
    return bl_cond, dl_cond


def _bl_cap_oracle(macro: Ye2023JsscCimMacro, x: Tensor) -> float:
    """Independent ``.bl_cap`` [fJ]: ``V_BL^2 * C_column`` per input-high physical column."""
    cfg = macro.config
    array_cfg = cfg.array_config
    cell_cfg = cfg.cell_config
    output_num = macro.col_num
    c_column__fF = (
        array_cfg.bl_first_c__fF
        + (output_num - 1) * array_cfg.bl_segment_c__fF
        + output_num * (cell_cfg.c_bl__fF + cell_cfg.c_x__fF)
    )
    _v_wl, _bl_v_ref, x_tiled = _drive(macro, x)
    high_col_count = float((x_tiled > 0).sum())
    return cfg.v_bl_in1__V**2 * c_column__fF * high_col_count


def _array_wl_only__fJ(macro: Ye2023JsscCimMacro) -> float:
    """Closed-form per-access array caps with every column input-low (WL side only)."""
    array_cfg = macro.config.array_config
    phys_col_num = macro.array.weight_grid_shape[-2]
    c_wl_wire__fF = array_cfg.wl_first_c__fF + (phys_col_num - 1) * array_cfg.wl_segment_c__fF
    per_access__fJ = (c_wl_wire__fF + phys_col_num * macro.config.cell_config.c_wl__fF) * V_WL_SEL__V**2
    return macro.col_num * per_access__fJ


def _stretched_window(config: Ye2023JsscCimMacroConfig, factor: float) -> Ye2023JsscCimMacroConfig:
    """The same config with every RS-CSA phase (and the latch delay) scaled."""
    adc = config.adc_config
    return dataclasses.replace(
        config,
        adc_config=dataclasses.replace(
            adc,
            t_phase__ns=tuple(factor * t for t in adc.t_phase__ns),
            t4_intrinsic__ns=factor * adc.t4_intrinsic__ns,
        ),
    )


def _wide_input_config(input_num: int) -> Ye2023JsscCimMacroConfig:
    """The witness widened to ``input_num`` logical inputs."""
    return dataclasses.replace(build_config(), max_active_num=input_num)


# ---------------------------------------------------------------------------
# (a) Channel / module presence + correct self-billing
# ---------------------------------------------------------------------------


def test_fig19_channels_present_and_self_billed(device: torch.device) -> None:
    """Every block appears under its exact name, positive, billed by the right module."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    macro, _prof, report = _run(build_config(), w, x, device=device)
    by_name = report.energy_by_name

    for key in _FIG19_KEYS:
        assert key in by_name, f"missing energy block {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive energy block {key!r}: {by_name[key]}"

    # The array self-bills its per-access caps as ONE un-channelled event.
    array_events = [e for e in report.energy_events if e.module is macro.array]
    assert len(array_events) == 1, f"array must bill once; got {len(array_events)}"
    assert array_events[0].channel is None and array_events[0].dynamic_energy__fJ > 0.0

    # The RS-CSA self-bills its conversion energy as ONE un-channelled event.
    rscsa_events = [e for e in report.energy_events if e.module is macro.rscsa]
    assert len(rscsa_events) == 1, f"rscsa must bill once; got {len(rscsa_events)}"
    assert rscsa_events[0].channel is None and rscsa_events[0].dynamic_energy__fJ > 0.0

    # The five channels are the MACRO ROOT's own events (branch-ownership law).
    macro_channels = {e.channel for e in report.energy_events if e.module is macro}
    assert macro_channels == set(_MACRO_CHANNELS), macro_channels

    # The cell is a non-reporter: it never self-bills a row.
    assert "cell" not in by_name, f"unexpected self-billing cell row: {sorted(by_name)}"

    # The rows partition the total: nothing is billed outside them.
    assert sum(by_name.values()) == pytest.approx(report.total_dynamic_energy__fJ)
    assert sum(by_name[key] for key in _FIG19_KEYS) == pytest.approx(report.total_dynamic_energy__fJ)


# ---------------------------------------------------------------------------
# (b) Branch ownership: conduction reconciles against a re-solve
# ---------------------------------------------------------------------------


def test_conduction_channels_reconcile_with_resolve(device: torch.device) -> None:
    """``.bl_cond`` / ``.dl_cond`` match an independent re-solve over the derived window, EXACTLY."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.long, device=device)  # batch (3,)
    macro, _prof, report = _run(build_config(), w, x, device=device)
    by_name = report.energy_by_name

    bl_cond, dl_cond = _conduction_oracle(macro, x.to(device))
    assert bl_cond > 0.0, f"witness draws no BL branch current: {bl_cond}"
    assert dl_cond > 0.0, f"witness draws no DL branch current: {dl_cond}"
    assert by_name[".bl_cond"] == pytest.approx(bl_cond)
    assert by_name[".dl_cond"] == pytest.approx(dl_cond)


def test_conduction_rides_the_derived_window(device: torch.device) -> None:
    """Stretching the phase set stretches BOTH conduction channels; caps do not move."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    base_cfg = build_config()
    macro, _p0, rep_base = _run(base_cfg, w, x, device=device)
    _m1, _p1, rep_wide = _run(_stretched_window(base_cfg, 2.0), w, x, device=device)
    b0, b1 = rep_base.energy_by_name, rep_wide.energy_by_name

    assert float(macro.t_ac__ns) == pytest.approx(
        sum(base_cfg.adc_config.t_phase__ns[:-1]) + base_cfg.adc_config.t4_intrinsic__ns
    )
    for channel in (".bl_cond", ".dl_cond"):
        assert b1[channel] == pytest.approx(2.0 * b0[channel]), f"{channel} does not ride T_AC"
    # The capacitive rows are window-INVARIANT: a conduction term left in either
    # would move it with the window and double-count the branch.
    for row in (_ARRAY_CAPS, ".bl_cap"):
        assert b0[row] > 0.0
        assert b1[row] == pytest.approx(b0[row]), f"{row} moved with the access window"


def test_conduction_and_latency_follow_the_executed_window(device: torch.device) -> None:
    """Lowering ``adc_bits`` shortens the window every conduction branch rides.

    The readout runs PH0 plus one compare phase per requested bit, so both
    conduction channels and the access latency scale by the executed-window
    ratio, while the capacitive rows and the per-op control lumps stay put. At
    the maximum resolution the ratio is the identity and the billed values are
    the nominal ones.
    """
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)

    macro, prof_full, rep_full = _run(cfg, w, x, device=device, adc_bits=TINY_ADC_BITS)
    full__fJ = rep_full.energy_by_name
    # The full-resolution run bills the NOMINAL window the macro publishes.
    assert prof_full.total_latency__ns == pytest.approx(float(macro.t_ac__ns) * TINY_OUTPUT_NUM)

    for bits in range(1, TINY_ADC_BITS + 1):
        macro_b, prof_b, rep_b = _run(cfg, w, x, device=device, adc_bits=bits)
        by_name = rep_b.energy_by_name
        ratio = float(macro_b.rscsa.t_conversion__ns(bits)) / float(macro_b.t_ac__ns)
        assert ratio <= 1.0 and (ratio < 1.0) == (bits < TINY_ADC_BITS), f"window ratio {ratio} at bits={bits}"
        for channel in (".bl_cond", ".dl_cond"):
            assert by_name[channel] == pytest.approx(ratio * full__fJ[channel]), f"{channel} at bits={bits}"
        assert prof_b.total_latency__ns == pytest.approx(ratio * prof_full.total_latency__ns)
        # Window-invariant rows: the caps ride no window, the control lumps are
        # per-op constants.
        for row in (_ARRAY_CAPS, ".bl_cap", ".mux_driver", ".timing_ctrl"):
            assert by_name[row] == pytest.approx(full__fJ[row]), f"{row} moved with the executed window"


def test_rscsa_zero_residue_row_is_the_prorated_baseline(device: torch.device) -> None:
    """At zero MAC the readout bills the code-independent baseline, prorated by the window.

    A zero-input access lands exactly on the derived PH0 compensation, so every
    compare phase weighs a zero residue and the conversion energy isolates the
    apportioned ``E_fixed``.
    """
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device)

    for bits in range(1, TINY_ADC_BITS + 1):
        macro, _prof, report = _run(cfg, w, x, device=device, adc_bits=bits)
        ratio = float(macro.rscsa.t_conversion__ns(bits)) / float(macro.t_ac__ns)
        expected__fJ = TINY_OUTPUT_NUM * cfg.adc_config.e_fixed_per_op__fJ * ratio
        assert report.energy_by_name[_RSCSA] == pytest.approx(expected__fJ), f"bits={bits}"


# ---------------------------------------------------------------------------
# (c) Three-part cap split
# ---------------------------------------------------------------------------


def test_bl_cap_is_the_per_vector_column_charge(device: torch.device) -> None:
    """``.bl_cap`` == ``V_BL^2 * C_column`` per input-high physical column, once per vector."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.tensor([[1, 1], [1, 0]], dtype=torch.long, device=device)  # batch (2,)
    macro, _prof, report = _run(build_config(), w, x, device=device)
    assert report.energy_by_name[".bl_cap"] == pytest.approx(_bl_cap_oracle(macro, x.to(device)))


def test_bl_cap_is_linear_in_active_inputs_and_zero_at_rest(device: torch.device) -> None:
    """The per-vector BL charge tracks the active-input count linearly and vanishes at zero input."""
    input_num = 4
    cfg = _wide_input_config(input_num)
    w = torch.full((input_num, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    def bl_cap(active: int) -> float:
        x = torch.zeros(input_num, dtype=torch.long, device=device)
        x[:active] = 1
        _m, _p, report = _run(cfg, w, x, device=device)
        return report.energy_by_name.get(".bl_cap", 0.0)

    samples = [bl_cap(k) for k in range(input_num + 1)]
    assert samples[0] == 0.0, f"idle BL charge billed: {samples[0]}"
    assert samples[1] > 0.0
    for k in range(1, input_num + 1):
        assert samples[k] == pytest.approx(k * samples[1]), f"bl_cap nonlinear at k={k}: {samples}"


def test_cap_ownership_split_between_array_and_macro(device: torch.device) -> None:
    """The array owns WL-side caps only; the BL / SL capacitances land on ``.bl_cap`` or nowhere."""
    base = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    _m, _p, rep_base = _run(base, w, x, device=device)
    b0 = rep_base.energy_by_name

    def with_array(array_config: Ye2023Jssc2t1rArrayConfig) -> Ye2023JsscCimMacroConfig:
        return dataclasses.replace(base, array_config=array_config)

    # SL capacitance is billed NOWHERE (the rail is grounded).
    sl_wire = with_array(dataclasses.replace(base.array_config, sl_first_c__fF=4.0, sl_segment_c__fF=1.0))
    sl_cell = with_array(
        dataclasses.replace(base.array_config, cell_config=dataclasses.replace(cell_config(), c_sl__fF=10.0))
    )
    for cfg in (sl_wire, sl_cell):
        _m, _p, rep = _run(cfg, w, x, device=device)
        assert rep.energy_by_name[_ARRAY_CAPS] == pytest.approx(b0[_ARRAY_CAPS])
        assert rep.energy_by_name[".bl_cap"] == pytest.approx(b0[".bl_cap"])

    # BL wire + BL node capacitance move the PER-VECTOR channel only.
    bl_wire = with_array(dataclasses.replace(base.array_config, bl_first_c__fF=4.0, bl_segment_c__fF=1.0))
    bl_cell = with_array(
        dataclasses.replace(base.array_config, cell_config=dataclasses.replace(cell_config(), c_bl__fF=10.0))
    )
    for cfg in (bl_wire, bl_cell):
        _m, _p, rep = _run(cfg, w, x, device=device)
        assert rep.energy_by_name[_ARRAY_CAPS] == pytest.approx(b0[_ARRAY_CAPS])
        assert rep.energy_by_name[".bl_cap"] > b0[".bl_cap"]

    # The WL side is the array's alone.
    wl_wire = with_array(dataclasses.replace(base.array_config, wl_first_c__fF=4.0, wl_segment_c__fF=1.0))
    _m, _p, rep_wl = _run(wl_wire, w, x, device=device)
    assert rep_wl.energy_by_name[_ARRAY_CAPS] > b0[_ARRAY_CAPS]
    assert rep_wl.energy_by_name[".bl_cap"] == pytest.approx(b0[".bl_cap"])


# ---------------------------------------------------------------------------
# (d) The all-off floor
# ---------------------------------------------------------------------------


def test_zero_input_bills_only_the_leakage_floor(device: torch.device) -> None:
    """At zero input the DL branch bills the derived PH0 current per output; BL bills nothing."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device)
    macro, _prof, report = _run(build_config(), w, x, device=device)
    by_name = report.energy_by_name

    t_ac__ns = float(macro.t_ac__ns)
    expected_dl = macro.config.v_dd_core__V * expected_ph0__uA() * TINY_OUTPUT_NUM * t_ac__ns
    assert by_name[".dl_cond"] == pytest.approx(expected_dl)
    assert macro.rscsa.i_ph0_comp__uA == pytest.approx(expected_ph0__uA())
    assert by_name.get(".bl_cond", 0.0) == 0.0
    assert by_name.get(".bl_cap", 0.0) == 0.0
    # The array collapses to its closed-form WL-only per-access value.
    assert by_name[_ARRAY_CAPS] == pytest.approx(_array_wl_only__fJ(macro))


def test_conduction_grows_with_active_inputs(device: torch.device) -> None:
    """Both conduction channels increase monotonically with the active-input count."""
    input_num = 4
    cfg = _wide_input_config(input_num)
    w = torch.full((input_num, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    def channels(active: int) -> tuple[float, float]:
        x = torch.zeros(input_num, dtype=torch.long, device=device)
        x[:active] = 1
        _m, _p, report = _run(cfg, w, x, device=device)
        by = report.energy_by_name
        return by.get(".bl_cond", 0.0), by.get(".dl_cond", 0.0)

    samples = [channels(k) for k in range(input_num + 1)]
    bl = [b for b, _ in samples]
    dl = [d for _, d in samples]
    assert bl[0] == 0.0
    assert all(lo < hi for lo, hi in itertools.pairwise(bl)), f"bl_cond not monotone: {bl}"
    assert all(lo < hi for lo, hi in itertools.pairwise(dl)), f"dl_cond not monotone: {dl}"
    # The BL branch is LINEAR in the active-input count (identical divider columns).
    for k in range(1, input_num + 1):
        assert bl[k] == pytest.approx(k * bl[1], rel=1e-3), f"bl_cond nonlinear at k={k}: {bl[k]}"


# ---------------------------------------------------------------------------
# (e) RS-CSA is E_fixed-dominant / flat
# ---------------------------------------------------------------------------


def test_rscsa_energy_flat_when_e_fixed_dominant(device: torch.device) -> None:
    """With ``e_fixed`` dominant the RS-CSA per-conversion energy is ~code-independent.

    All four outputs carry the SAME code, so the row is
    ``output_num * (E_fixed + E_code(code))``: a zero residue costs exactly
    ``E_fixed``, every code stays inside the all-reference envelope
    ``sum_p k * v_rail * t_phase_p * I_REF_p``, and the sweep spread measures the
    data-dependent share alone. The per-phase reference is the ONE injected
    current at that phase's binary place value, ``2**(bits - p) * i_ref``.
    """
    base = build_config()
    adc = dataclasses.replace(base.adc_config, e_fixed_per_op__fJ=1.0e4)
    cfg = dataclasses.replace(base, adc_config=adc)
    i_ref__uA = cfg.reference_config.i_refs__uA[QUANTIZATION_MODE][0]

    def rscsa_energy(w_in0: int, w_in1: int) -> float:
        w = torch.zeros((TINY_INPUT_NUM, TINY_OUTPUT_NUM), dtype=torch.long, device=device)
        w[0, :] = w_in0
        w[1, :] = w_in1
        _m, _p, report = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
        return report.energy_by_name[_RSCSA]

    # Codes 0 (no residue), 7 (mid bits), 14 (MSB latched) — spanning the ladder.
    energies = [rscsa_energy(0, 0), rscsa_energy(7, 0), rscsa_energy(7, 7)]
    floor__fJ = TINY_OUTPUT_NUM * adc.e_fixed_per_op__fJ
    envelope__fJ = floor__fJ + TINY_OUTPUT_NUM * sum(
        adc.mirror_scale * adc.v_rail__V * adc.t_phase__ns[phase] * (1 << (adc.bits - phase)) * i_ref__uA
        for phase in range(1, adc.bits + 1)
    )
    # A zero-MAC conversion leaves no residue to compare: E_fixed alone.
    assert energies[0] == pytest.approx(floor__fJ)
    for e in energies:
        assert floor__fJ <= e <= envelope__fJ, f"conversion energy outside [E_fixed, envelope]: {e}"
    spread = (max(energies) - min(energies)) / min(energies)
    assert spread < 0.05, f"RS-CSA not flat under dominant e_fixed: {energies} (spread {spread:.4f})"


# ---------------------------------------------------------------------------
# (f) Latency is the macro's sole emission
# ---------------------------------------------------------------------------


def test_latency_is_macro_sole_t_ac_times_serial(device: torch.device) -> None:
    """Exactly one latency event of ``T_AC * serial_rounds``; the array / RS-CSA emit none."""
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    # Single access: one serial conversion per output.
    macro, prof, _rep = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
    t_ac = float(macro.t_ac__ns)
    assert len(prof.latency_events) == 1, f"macro must be the sole latency emitter; got {len(prof.latency_events)}"
    assert prof.latency_events[0].module is macro
    assert prof.total_latency__ns == pytest.approx(t_ac * TINY_OUTPUT_NUM)

    # A batch multiplies the serial round count (numel / inst_count).
    x_batch = torch.ones((3, TINY_INPUT_NUM), dtype=torch.long, device=device)
    _m, prof_b, _r = _run(cfg, w, x_batch, device=device)
    assert len(prof_b.latency_events) == 1
    assert prof_b.total_latency__ns == pytest.approx(t_ac * TINY_OUTPUT_NUM * 3)


# ---------------------------------------------------------------------------
# (g) Static leakage reconciles via collect_static
# ---------------------------------------------------------------------------


def test_static_leakage_reconciles_via_collect_static(device: torch.device) -> None:
    """``leakage_energy == sum(collect_static leakage) * total_latency``; scales with serial rounds."""
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    macro, prof, report = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
    seats = {r.qualified_name: r.leakage_power__uW for r in NeuroxProfiler.collect_static(macro)}
    for seat in ("", "array", "rscsa", "bl_driver", "sl_driver", "mux_driver", "timing_ctrl"):
        assert seats.get(seat, 0.0) > 0.0, f"missing/empty static seat {seat!r}; have {sorted(seats)}"

    total_leakage = sum(seats.values())
    assert report.static.leakage_power__uW == pytest.approx(total_leakage)
    # leakage_energy = leakage_power * total_latency (the macro is the sole latency emitter).
    assert report.leakage_energy__fJ == pytest.approx(total_leakage * prof.total_latency__ns)

    # A batch scales the latency (serial rounds) -> the leakage ENERGY scales; the
    # per-seat leakage POWER does not.
    _m, _prof_b, report_b = _run(
        cfg,
        w,
        torch.ones((3, TINY_INPUT_NUM), dtype=torch.long, device=device),
        device=device,
    )
    assert report_b.static.leakage_power__uW == pytest.approx(total_leakage)
    assert report_b.leakage_energy__fJ == pytest.approx(3.0 * report.leakage_energy__fJ)


def test_bl_conduction_rides_the_input_rail(device: torch.device) -> None:
    """``.bl_cond`` is billed across ``v_bl_in1__V``, not the core rail."""
    base = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    _m, _p, rep_base = _run(base, w, x, device=device)

    # Raising the core rail leaves the BL branch untouched and scales DL. The
    # readout mirror rides the same rail, so both move together.
    hot_core = dataclasses.replace(
        base,
        v_dd_core__V=2.0 * base.v_dd_core__V,
        adc_config=dataclasses.replace(base.adc_config, v_rail__V=2.0 * base.adc_config.v_rail__V),
    )
    _m, _p, rep_hot = _run(hot_core, w, x, device=device)
    assert rep_hot.energy_by_name[".bl_cond"] == pytest.approx(rep_base.energy_by_name[".bl_cond"])
    assert rep_hot.energy_by_name[".dl_cond"] == pytest.approx(2.0 * rep_base.energy_by_name[".dl_cond"])
    assert base.v_bl_in1__V == V_BL_IN1__V
