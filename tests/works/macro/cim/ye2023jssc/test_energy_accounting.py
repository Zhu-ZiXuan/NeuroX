"""Eager energy-accounting laws for the ye2023jssc WH-2T1R CIM macro.

Pins the Fig.19 branch-ownership and channel-billing contract on the hand-built
analytic witness (``_utils.build_config``). Every check is a LAW read off the
profiler report: the coefficients are whatever the deterministic all-off analog
chain produces, and the assertions constrain how the billed energy MOVES.

Coverage:

  * the Fig.19 blocks appear under their exact channel / module names — the
    per-access array caps row ``array``, the two converter rows ``wl_dac`` /
    ``bl_dac``, the macro-owned ``.bl_cond`` / ``.dl_cond`` conduction channels,
    the RS-CSA row ``rscsa``, and the two flat peripheral channels
    ``.mux_driver`` / ``.timing_ctrl`` — all positive, each billed by the RIGHT
    module (array / converters / RS-CSA self-bill un-channelled; the four
    channels are the macro root's own events; no ``cell`` self-bill row leaks
    through), and they ADD UP to the report's total dynamic energy,
  * branch ownership: both conduction channels reconcile EXACTLY against an
    independent re-solve oracle over ONE window — ``.bl_cond == V_DD_bl *
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
  * caps are the ARRAY's alone: it bills every per-node total inside itself under
    the kernel's held-BL scan law (WL node off the gate drive, BL and X nodes off
    the held rest level, SL node off its grounded rest), and the mode contract
    that licenses its hold amortization holds here — one call drives one BL
    pattern and scans every array row exactly once,
  * the converters bill their own drive events off their per-code tables: the WL
    bank once per word line per access (one selected code, every other line
    deselected) and the BL bank once per physical column per input VECTOR, each
    code paying its own entry,
  * the all-off floor: with zero input the DL branch bills exactly the derived
    PH0 current on every output, the BL conduction vanishes, the BL converter
    still bills its code-0 entry per column, and the array collapses to its
    closed-form WL-only value,
  * the RS-CSA is E_fixed-dominant: when ``e_fixed`` dwarfs the per-code SAR
    energy the per-conversion energy is code-independent to within a few percent,
  * latency is ``T_AC`` over the macro's own output axis, invariant to a caller
    batch,
  * static leakage reconciles: ``static.leakage__uW`` is the sum over the
    reporter's static entries, and a batch does not move it.

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

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog.voltage_dac import GeneralVdacConfig
from neurox.works.macro.cim.ye2023jssc.array import Ye2023Jssc2t1rArrayConfig

from ._utils import (
    DTYPE,
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    V_WL_SEL__V,
    W_MAX,
    E_BL_DAC_PER_CODE__fJ,
    E_WL_DAC_PER_CODE__fJ,
    Ye2023JsscCimMacro,
    Ye2023JsscCimMacroConfig,
    build_config,
    build_macro,
    expected_ph0__uA,
)

# The Fig.19 blocks as they surface in the reporter's row names (macro root == "").
_ARRAY_CAPS = "array"
_WL_DAC = "wl_dac"
_BL_DAC = "bl_dac"
_RSCSA = "rscsa"
_MACRO_CHANNELS = ("bl_cond", "dl_cond", "mux_driver", "timing_ctrl")
_FIG19_KEYS = (
    _ARRAY_CAPS,
    _WL_DAC,
    _BL_DAC,
    ".bl_cond",
    ".dl_cond",
    _RSCSA,
    ".mux_driver",
    ".timing_ctrl",
)


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
) -> tuple[Ye2023JsscCimMacro, Profiler, Reporter]:
    """Build + fabricate a fresh macro, program ``w``, profile one VMM on ``x``."""
    macro = build_macro(config, input_num=w.shape[-2], output_num=w.shape[-1], device=device)
    stamp_names(macro)
    macro.program(w.to(device))
    with Profiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=adc_bits)
    return macro, prof, Reporter(macro)


def _dac_levels(config: Ye2023JsscCimMacroConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The ``(wl, bl)`` code-to-level tables the two converter banks drive from."""
    wl_dac_config = config.wl_dac_config
    bl_dac_config = config.bl_dac_config
    assert isinstance(wl_dac_config, GeneralVdacConfig)
    assert isinstance(bl_dac_config, GeneralVdacConfig)
    return wl_dac_config.code_to_signal, bl_dac_config.code_to_signal


def _drive(macro: Ye2023JsscCimMacro, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Rebuild the exact drive ``vec_mat_mul`` constructs, from the config alone.

    Returns the full-grid word-line drive (one value per cell gate, one-hot over
    the array rows and flat across the columns), the per-column BL reference
    voltages broadcast over the output leading, and the plane-major input
    indicator (weight planes tiled, redundant planes forced input-0).
    """
    cfg = macro.config
    device = x.device
    n_weight_plane = cfg.w_digit_num
    n_redundant_plane = len(cfg.array_config.redundant_radix)
    input_num = macro.row_num
    output_num = macro.col_num
    wl_levels__V, bl_levels__V = _dac_levels(cfg)

    x_long = x.long()
    leading = x_long.shape[:-1]
    x_weight = (
        x_long.unsqueeze(-2).expand(*leading, n_weight_plane, input_num).reshape(*leading, n_weight_plane * input_num)
    )
    x_tiled = torch.cat((x_weight, x_weight.new_zeros((*leading, n_redundant_plane * input_num))), dim=-1)
    v_bl = torch.where(
        x_tiled > 0,
        torch.tensor(bl_levels__V[1], dtype=DTYPE, device=device),
        torch.tensor(bl_levels__V[0], dtype=DTYPE, device=device),
    )
    phys_col_num = x_tiled.shape[-1]
    selected = torch.eye(output_num, dtype=torch.bool, device=device)
    wl_onehot = torch.where(
        selected,
        torch.tensor(wl_levels__V[1], dtype=DTYPE, device=device),
        torch.tensor(wl_levels__V[0], dtype=DTYPE, device=device),
    )
    # Shape: [..., out, phys_col, array_row]
    v_wl = wl_onehot.unsqueeze(-2).expand(*leading, output_num, phys_col_num, output_num)
    bl_v_ref = v_bl.unsqueeze(-2).expand(*leading, output_num, phys_col_num)
    return v_wl, bl_v_ref, x_tiled


def _conduction_oracle(macro: Ye2023JsscCimMacro, x: Tensor) -> tuple[float, float]:
    """Independent ``(.bl_cond, .dl_cond)`` [fJ] from a re-solve over the derived window.

    Re-runs the array DC solve OUTSIDE any profiler (so it logs nothing of its
    own) and applies the ONE access window every conduction branch rides. Each
    branch is billed on the rail its charge leaves, never on the node level it
    feeds::

        bl_cond = V_DD_bl   * sum(I_BL_port) * T_AC
        dl_cond = V_DD_core * sum(I_TBL_raw) * T_AC
    """
    cfg = macro.config
    v_wl, bl_v_ref, _x_tiled = _drive(macro, x)
    # The caller owns the event structure, so the snaps are taken here too.
    event_shape = tuple(bl_v_ref.shape)
    v_sl__V = torch.tensor(cfg.v_sl__V, dtype=DTYPE, device=x.device)
    steady = macro.array.solve_array(
        v_wl,
        bl_driver=macro.bl_driver,
        bl_driver_snap=macro.bl_driver.snapshot(v_ref__V=bl_v_ref, shape=event_shape),
        sl_driver=macro.sl_driver,
        sl_driver_snap=macro.sl_driver.snapshot(v_ref__V=v_sl__V.expand(event_shape), shape=event_shape),
    )
    t_ac__ns = float(macro.t_ac__ns)
    bl_cond = float((cfg.v_dd_bl__V * steady.i_bl_port__uA).sum() * t_ac__ns)
    dl_cond = float((cfg.v_dd_core__V * steady.i_tbl__uA).sum() * t_ac__ns)
    return bl_cond, dl_cond


def _wl_dac_oracle__fJ(macro: Ye2023JsscCimMacro) -> float:
    """Independent ``wl_dac`` [fJ]: one drive event per word line per output access.

    Every access raises ONE line and holds the other ``col_num - 1`` at their
    deselected code, and each code pays its own table entry.
    """
    output_num = macro.col_num
    selected__fJ, deselected__fJ = E_WL_DAC_PER_CODE__fJ[1], E_WL_DAC_PER_CODE__fJ[0]
    return output_num * (selected__fJ + (output_num - 1) * deselected__fJ)


def _bl_dac_oracle__fJ(macro: Ye2023JsscCimMacro, x: Tensor) -> float:
    """Independent ``bl_dac`` [fJ]: one drive event per physical column per input vector."""
    _v_wl, _bl_v_ref, x_tiled = _drive(macro, x)
    high = float((x_tiled > 0).sum())
    low = float(x_tiled.numel()) - high
    return high * E_BL_DAC_PER_CODE__fJ[1] + low * E_BL_DAC_PER_CODE__fJ[0]


def _array_wl_only__fJ(macro: Ye2023JsscCimMacro) -> float:
    """Closed-form array caps with every column input-low: the WL side alone.

    A zero-input access holds nothing, so both rails rest at and settle to 0 V
    and the supply-draw law leaves each cell's WL node total (gate load plus that
    node's line share), ``V_DD_WL * C * |V_WL_sel|`` per driven gate.
    """
    cfg = macro.config
    phys_col_num = macro.array.weight_grid_shape[-2]
    per_access__fJ = phys_col_num * cfg.array_config.wl_node_c__fF * cfg.v_dd_wl__V * V_WL_SEL__V
    return macro.col_num * per_access__fJ


def _stretched_window(config: Ye2023JsscCimMacroConfig, factor: float) -> Ye2023JsscCimMacroConfig:
    """The same config with every RS-CSA phase (and every latch delay) scaled."""
    adc = config.adc_config
    return dataclasses.replace(
        config,
        adc_config=dataclasses.replace(
            adc,
            t_phase__ns=tuple(factor * t for t in adc.t_phase__ns),
            t_intrinsic__ns=tuple(factor * t for t in adc.t_intrinsic__ns),
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
    macro, prof, reporter = _run(build_config(), w, x, device=device)
    by_name = reporter.by_name(prof)

    for key in _FIG19_KEYS:
        assert key in by_name, f"missing energy block {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive energy block {key!r}: {by_name[key]}"

    # The array self-bills its per-access caps as ONE un-channelled record.
    array_records = [r for r in prof.records if r.qualified_name == macro.array.qualified_name]
    assert len(array_records) == 1, f"array must bill once; got {len(array_records)}"
    assert array_records[0].channel is None and array_records[0].dynamic_energy__fJ > 0.0

    # The RS-CSA self-bills its conversion energy as ONE un-channelled record.
    rscsa_records = [r for r in prof.records if r.qualified_name == macro.rscsa.qualified_name]
    assert len(rscsa_records) == 1, f"rscsa must bill once; got {len(rscsa_records)}"
    assert rscsa_records[0].channel is None and rscsa_records[0].dynamic_energy__fJ > 0.0

    # Each converter bank self-bills its drive as ONE un-channelled record.
    for bank in (macro.wl_dac, macro.bl_dac):
        dac_records = [r for r in prof.records if r.qualified_name == bank.qualified_name]
        assert len(dac_records) == 1, f"{type(bank).__name__} must bill once; got {len(dac_records)}"
        assert dac_records[0].channel is None and dac_records[0].dynamic_energy__fJ > 0.0

    # The four channels are the MACRO ROOT's own records (branch-ownership law).
    macro_channels = {r.channel for r in prof.records if r.qualified_name == macro.qualified_name}
    assert macro_channels == set(_MACRO_CHANNELS), macro_channels

    # The cell is a non-reporter: it never self-bills a row.
    assert "cell" not in by_name, f"unexpected self-billing cell row: {sorted(by_name)}"

    # The rows partition the total: nothing is billed outside them.
    total__fJ = reporter.total_dynamic_energy__fJ(prof)
    assert sum(by_name.values()) == pytest.approx(total__fJ)
    assert sum(by_name[key] for key in _FIG19_KEYS) == pytest.approx(total__fJ)


# ---------------------------------------------------------------------------
# (b) Branch ownership: conduction reconciles against a re-solve
# ---------------------------------------------------------------------------


def test_conduction_channels_reconcile_with_resolve(device: torch.device) -> None:
    """``.bl_cond`` / ``.dl_cond`` match an independent re-solve over the derived window, EXACTLY."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.long, device=device)  # batch (3,)
    macro, prof, reporter = _run(build_config(), w, x, device=device)
    by_name = reporter.by_name(prof)

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
    macro, p0, rep_base = _run(base_cfg, w, x, device=device)
    _m1, p1, rep_wide = _run(_stretched_window(base_cfg, 2.0), w, x, device=device)
    b0, b1 = rep_base.by_name(p0), rep_wide.by_name(p1)

    assert float(macro.t_ac__ns) == pytest.approx(
        sum(base_cfg.adc_config.t_phase__ns[:-1]) + base_cfg.adc_config.t_intrinsic__ns[-1]
    )
    for channel in (".bl_cond", ".dl_cond"):
        assert b1[channel] == pytest.approx(2.0 * b0[channel]), f"{channel} does not ride T_AC"
    # The capacitive and drive rows are window-INVARIANT: a conduction term left
    # in any of them would move it with the window and double-count the branch.
    for row in (_ARRAY_CAPS, _WL_DAC, _BL_DAC):
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
    full__fJ = rep_full.by_name(prof_full)
    # The full-resolution run reports the NOMINAL window the macro publishes.
    full_latency__ns = macro.latency__ns(adc_bits=TINY_ADC_BITS)
    assert full_latency__ns == pytest.approx(float(macro.t_ac__ns) * TINY_OUTPUT_NUM)

    for bits in range(1, TINY_ADC_BITS + 1):
        macro_b, prof_b, rep_b = _run(cfg, w, x, device=device, adc_bits=bits)
        by_name = rep_b.by_name(prof_b)
        ratio = float(macro_b.rscsa.t_conversion__ns(bits)) / float(macro_b.t_ac__ns)
        assert ratio <= 1.0 and (ratio < 1.0) == (bits < TINY_ADC_BITS), f"window ratio {ratio} at bits={bits}"
        for channel in (".bl_cond", ".dl_cond"):
            assert by_name[channel] == pytest.approx(ratio * full__fJ[channel]), f"{channel} at bits={bits}"
        assert macro_b.latency__ns(adc_bits=bits) == pytest.approx(ratio * full_latency__ns)
        # Window-invariant rows: the caps and the drive events ride no window,
        # the control lumps are per-op constants.
        for row in (_ARRAY_CAPS, _WL_DAC, _BL_DAC, ".mux_driver", ".timing_ctrl"):
            assert by_name[row] == pytest.approx(full__fJ[row]), f"{row} moved with the executed window"


def test_rscsa_zero_residue_row_is_the_prorated_baseline(device: torch.device) -> None:
    """At zero MAC the readout bills the code-independent baseline, prorated by the window.

    A zero-input access lands exactly on the seated PH0 compensation, so every
    compare phase weighs a zero residue and the conversion energy isolates the
    apportioned ``E_fixed``.
    """
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device)

    for bits in range(1, TINY_ADC_BITS + 1):
        macro, prof, reporter = _run(cfg, w, x, device=device, adc_bits=bits)
        ratio = float(macro.rscsa.t_conversion__ns(bits)) / float(macro.t_ac__ns)
        expected__fJ = TINY_OUTPUT_NUM * cfg.adc_config.e_fixed_per_op__fJ * ratio
        assert reporter.by_name(prof)[_RSCSA] == pytest.approx(expected__fJ), f"bits={bits}"


# ---------------------------------------------------------------------------
# (c) Cap ownership and the two converter drives
# ---------------------------------------------------------------------------


def test_bl_dac_bills_one_drive_event_per_column_per_vector(device: torch.device) -> None:
    """``bl_dac`` == the per-code table summed over every column of every input vector."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.tensor([[1, 1], [1, 0]], dtype=torch.long, device=device)  # batch (2,)
    macro, prof, reporter = _run(build_config(), w, x, device=device)
    assert reporter.by_name(prof)[_BL_DAC] == pytest.approx(_bl_dac_oracle__fJ(macro, x.to(device)))


def test_bl_dac_tracks_the_active_input_count_by_its_own_code_step(device: torch.device) -> None:
    """Raising one more input swaps one column's code, so the row moves by ONE code step.

    The level is held across the row scan, so the swap is billed once per vector,
    once per plane the input is tiled onto — never per access.
    """
    input_num = 4
    cfg = _wide_input_config(input_num)
    w = torch.full((input_num, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    def bl_dac(active: int) -> float:
        x = torch.zeros(input_num, dtype=torch.long, device=device)
        x[:active] = 1
        _m, prof, reporter = _run(cfg, w, x, device=device)
        return reporter.by_name(prof)[_BL_DAC]

    samples = [bl_dac(k) for k in range(input_num + 1)]
    # An idle column is still driven: the code-0 entry is what it pays.
    assert samples[0] > 0.0
    step__fJ = cfg.w_digit_num * (E_BL_DAC_PER_CODE__fJ[1] - E_BL_DAC_PER_CODE__fJ[0])
    for k in range(input_num + 1):
        assert samples[k] == pytest.approx(samples[0] + k * step__fJ), f"bl_dac off its code step at k={k}: {samples}"


def test_wl_dac_bills_one_drive_event_per_line_per_access(device: torch.device) -> None:
    """``wl_dac`` == one selected code plus ``col_num - 1`` deselected ones, per access."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    macro, prof, reporter = _run(build_config(), w, x, device=device)
    assert reporter.by_name(prof)[_WL_DAC] == pytest.approx(_wl_dac_oracle__fJ(macro))


def test_array_owns_every_node_capacitance(device: torch.device) -> None:
    """Every per-node total moves the ``array`` row, and moves nothing else.

    Under the kernel's held-BL scan law the array bills its whole node set — the
    WL node off the gate drive, the BL and X nodes off the level they rest at,
    the SL node off the IR-drop displacement from its grounded rest. The array's
    node ledger is the sole account of that capacitance, so no other row follows
    it.
    """
    base = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    _m, p_base, rep_base = _run(base, w, x, device=device)
    b0 = rep_base.by_name(p_base)

    def with_array(array_config: Ye2023Jssc2t1rArrayConfig) -> Ye2023JsscCimMacroConfig:
        return dataclasses.replace(base, array_config=array_config)

    for field in ("bl_node_c__fF", "x_node_c__fF", "sl_node_c__fF", "wl_node_c__fF"):
        heavier = with_array(dataclasses.replace(base.array_config, **{field: 10.0}))
        _m, prof, rep = _run(heavier, w, x, device=device)
        by_name = rep.by_name(prof)
        assert by_name[_ARRAY_CAPS] > b0[_ARRAY_CAPS], f"array blind to {field}"
        for row in (_WL_DAC, _BL_DAC, ".bl_cond", ".dl_cond"):
            assert by_name[row] == pytest.approx(b0[row]), f"{row} follows {field}"


def test_one_hold_covers_exactly_one_row_scan(device: torch.device) -> None:
    """The mode contract the array's precharge amortization rests on.

    ``BL_IN_WL_SCAN`` bills one hold establishment per ``row_num`` solves. This
    macro satisfies that exactly: one ``vec_mat_mul`` drives ONE BL input
    pattern and scans every array row once, because the output axis it
    serializes over IS the array's row axis.
    """
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    macro, _prof, _rep = _run(cfg, w, x, device=device)

    # Physical rows == logical outputs == the accesses one call serializes.
    array_row_num = macro.array.weight_grid_shape[-1]
    assert array_row_num == macro.col_num == TINY_OUTPUT_NUM
    # And the held pattern is the same for every one of those accesses: the BL
    # reference carries no output axis of its own before it is broadcast.
    _v_wl, bl_v_ref, _x_tiled = _drive(macro, x.to(device))
    assert bl_v_ref.shape[-2] == array_row_num
    assert torch.equal(bl_v_ref, bl_v_ref[..., :1, :].expand_as(bl_v_ref))


# ---------------------------------------------------------------------------
# (d) The all-off floor
# ---------------------------------------------------------------------------


def test_zero_input_bills_only_the_leakage_floor(device: torch.device) -> None:
    """At zero input the DL branch bills the seated PH0 current per output; BL conducts nothing."""
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device)
    macro, prof, reporter = _run(build_config(), w, x, device=device)
    by_name = reporter.by_name(prof)

    t_ac__ns = float(macro.t_ac__ns)
    expected_dl = macro.config.v_dd_core__V * expected_ph0__uA() * TINY_OUTPUT_NUM * t_ac__ns
    assert by_name[".dl_cond"] == pytest.approx(expected_dl)
    assert macro.rscsa.i_ph0_comp__uA == pytest.approx(expected_ph0__uA())
    assert by_name.get(".bl_cond", 0.0) == 0.0
    # A column driven to the IN = 0 level is still driven: the converter bills
    # that code's own entry on every physical column.
    phys_col_num = macro.array.weight_grid_shape[-2]
    assert by_name[_BL_DAC] == pytest.approx(phys_col_num * E_BL_DAC_PER_CODE__fJ[0])
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
        _m, prof, reporter = _run(cfg, w, x, device=device)
        by = reporter.by_name(prof)
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
        _m, prof, reporter = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
        return reporter.by_name(prof)[_RSCSA]

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
# (f) Latency is T_AC over the macro's own output axis
# ---------------------------------------------------------------------------


def test_latency_is_t_ac_over_the_output_axis(device: torch.device) -> None:
    """``T_AC`` per logical output — the one time axis the macro owns."""
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    macro, _prof, _rep = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
    t_ac = float(macro.t_ac__ns)
    assert macro.latency__ns(adc_bits=TINY_ADC_BITS) == pytest.approx(t_ac * TINY_OUTPUT_NUM)


# ---------------------------------------------------------------------------
# (g) Static leakage reconciles via the reporter's static entries
# ---------------------------------------------------------------------------


def test_static_leakage_reconciles_via_static_entries(device: torch.device) -> None:
    """``static.leakage__uW == sum(static entry leakage)``, batch-invariant."""
    cfg = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)

    _macro, _prof, reporter = _run(cfg, w, torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device), device=device)
    seats = {e.qualified_name: e.leakage__uW for e in reporter.static_entries}
    # The array's lattice rests at zero cell bias and holds no static conduction
    # path, so it seats area without leakage; every other seat leaks.
    assert "array" in seats
    for seat in ("", "rscsa", "bl_driver", "sl_driver", "mux_driver", "timing_ctrl"):
        assert seats.get(seat, 0.0) > 0.0, f"missing/empty static seat {seat!r}; have {sorted(seats)}"

    total_leakage = sum(seats.values())
    assert reporter.static.leakage__uW == pytest.approx(total_leakage)

    # Leakage POWER is fabrication-fixed: a batch does not move it.
    _m, _prof_b, reporter_b = _run(
        cfg,
        w,
        torch.ones((3, TINY_INPUT_NUM), dtype=torch.long, device=device),
        device=device,
    )
    assert reporter_b.static.leakage__uW == pytest.approx(total_leakage)


def test_bl_conduction_rides_the_bl_driver_rail(device: torch.device) -> None:
    """``.bl_cond`` is billed across ``v_dd_bl__V``, not the core rail.

    A branch bill states which supply the charge leaves. The two rails are
    separate variables even when numerically equal, so moving one must move its
    own branch and nothing else's.
    """
    base = build_config()
    w = torch.full((TINY_INPUT_NUM, TINY_OUTPUT_NUM), W_MAX, dtype=torch.long, device=device)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long, device=device)
    _m, p_base, rep_base = _run(base, w, x, device=device)
    base__fJ = rep_base.by_name(p_base)

    # Raising the core rail leaves the BL branch untouched and scales DL. The
    # readout mirror rides the same rail, so both move together.
    hot_core = dataclasses.replace(
        base,
        v_dd_core__V=2.0 * base.v_dd_core__V,
        adc_config=dataclasses.replace(base.adc_config, v_rail__V=2.0 * base.adc_config.v_rail__V),
    )
    _m, p_hot, rep_hot = _run(hot_core, w, x, device=device)
    hot__fJ = rep_hot.by_name(p_hot)
    assert hot__fJ[".bl_cond"] == pytest.approx(base__fJ[".bl_cond"])
    assert hot__fJ[".dl_cond"] == pytest.approx(2.0 * base__fJ[".dl_cond"])

    # Raising the BL driver rail scales the BL branch by that factor and leaves
    # the DL branch alone. The BL node levels are the converter's, so the solved
    # port current does not move with the rail.
    hot_bl = dataclasses.replace(base, v_dd_bl__V=2.0 * base.v_dd_bl__V)
    _m, p_bl, rep_bl = _run(hot_bl, w, x, device=device)
    bl__fJ = rep_bl.by_name(p_bl)
    assert bl__fJ[".bl_cond"] == pytest.approx(2.0 * base__fJ[".bl_cond"])
    assert bl__fJ[".dl_cond"] == pytest.approx(base__fJ[".dl_cond"])
