"""Xue2020 energy ownership and timing tests."""

from __future__ import annotations

import dataclasses
from typing import TypedDict

import pytest
import torch
from torch import Tensor

from neurox import Profiler, Reporter

from ._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    build_config,
    build_macro,
)


class _TmcsaConfigUpdates(TypedDict, total=False):
    t_ph2__ns: float
    t_ph3__ns: float
    energy_per_bit__fJ: float


# Billed rows by slice name -> profiler energy-row key.
_ROW_KEYS = {
    "cablc": ".cablc",  # macro channel
    "dswct": ".dswct",  # macro channel
    "sinwp_sc": ".sinwp_sc",  # macro channel
    "pn_isub": ".pn_isub",  # macro channel
    "control": "control",  # module row
}
_READ_ROWS = ("cablc", "dswct", "sinwp_sc")  # sampled-phase conduction rows


def _w_full(input_num: int = TINY_INPUT_NUM, output_num: int = TINY_OUTPUT_NUM) -> Tensor:
    """All-`+1` weights so every physical column and scan conducts."""
    return torch.ones((input_num, output_num), dtype=torch.long)


def _x_full(k: int, input_num: int = TINY_INPUT_NUM) -> Tensor:
    """Full-scale K-bit input (every input bit set on every row) — all sub-phase legs conduct."""
    return torch.full((input_num,), (1 << k) - 1, dtype=torch.long)


def _run(
    config: Xue2020JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    quantization_mode: int = QUANTIZATION_MODE,
    adc_active_bits: int = TINY_ADC_BITS,
) -> tuple[Profiler, Reporter]:
    """Build + fabricate a fresh macro, program `w`, profile one VMM on `x`."""
    macro = build_macro(config, device=device)
    macro.program(w.to(device))
    with Profiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
    return prof, Reporter(macro)


def _channels(by_name: dict[str, float]) -> dict[str, float]:
    """Per-row dynamic energy [fJ] keyed by slice name."""
    return {name: by_name.get(key, 0.0) for name, key in _ROW_KEYS.items()}


def _channel_energies(
    config: Xue2020JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    adc_active_bits: int = TINY_ADC_BITS,
) -> dict[str, float]:
    prof, reporter = _run(config, w, x, device=device, adc_active_bits=adc_active_bits)
    return _channels(reporter.by_name(prof))


def _with_tmcsa(
    config: Xue2020JsscCimMacroConfig,
    *,
    t_ph2__ns: float | None = None,
    t_ph3__ns: float | None = None,
    energy_per_bit__fJ: float | None = None,
) -> Xue2020JsscCimMacroConfig:
    """Replace only TMCSA energy-model parameters."""
    kw: _TmcsaConfigUpdates = {}
    if t_ph2__ns is not None:
        kw["t_ph2__ns"] = t_ph2__ns
    if t_ph3__ns is not None:
        kw["t_ph3__ns"] = t_ph3__ns
    if energy_per_bit__fJ is not None:
        kw["energy_per_bit__fJ"] = energy_per_bit__fJ
    return dataclasses.replace(config, tmcsa_config=dataclasses.replace(config.tmcsa_config, **kw))


def _whole_input_branch(macro: Xue2020JsscCimMacro, x: Tensor) -> float:
    """Re-solve and sum `VDD * I_DL * phase_duration` [fJ]."""
    cfg = macro.config
    vdd__V = cfg.vdd__V
    x_long = x.long()
    sample__ns = cfg.t_sample__ns
    detect__ns = cfg.t_settle__ns + macro.tmcsa.latency__ns(active_bits=TINY_ADC_BITS)

    planes = torch.stack(tuple((x_long >> k) & 1 for k in range(cfg.x_bit_num)), dim=-2)
    v_wl = planes * macro._v_wl_on__V
    leading = tuple(torch.broadcast_shapes(macro.inst_shape, v_wl.shape[:-1]))
    ref_shape = (*leading, 1, macro.lane_num, macro.scan_num, 2, cfg.w_digit_num)
    v_blc = macro.cablc_vref.values()
    dcop = macro.array.solve_dc(
        v_wl__V=v_wl.unsqueeze(macro.array.col_dim),
        wl_phase_dims=(-3,),
        bl_driver_snap=macro.cablc.snapshot(v_ref__V=v_blc.expand(ref_shape), shape=ref_shape).flatten_axes(-4, -1),
        sl_driver_snap=macro.sl_driver.snapshot(
            v_ref__V=torch.zeros((), dtype=v_wl.dtype, device=v_wl.device).expand(ref_shape),
            shape=ref_shape,
        ).flatten_axes(-4, -1),
    )
    phase_duration__ns = v_wl.new_tensor((*([sample__ns] * (cfg.x_bit_num - 1)), detect__ns))
    energy__fJ = (vdd__V * dcop.i_bl_port__uA).sum(dim=(macro.array.row_dim, macro.array.col_dim)) * phase_duration__ns
    return float(energy__fJ.sum())


def test_billed_rows_present_with_exact_names(device: torch.device) -> None:
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)
    prof, reporter = _run(build_config(), _w_full(), x, device=device)
    by_name = reporter.by_name(prof)
    for name, key in _ROW_KEYS.items():
        assert key in by_name, f"missing {name} row {key!r}; have {sorted(by_name)}"
        assert by_name[key] > 0.0, f"non-positive {key}: {by_name[key]}"
    for stale in ("dswct", "sinwp_sc", "pn_isub"):
        assert stale not in by_name, f"stale energy row {stale}: {sorted(by_name)}"


def test_channels_and_module_rows_bill_dynamic(device: torch.device) -> None:
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)
    prof, reporter = _run(build_config(), _w_full(), x, device=device)
    by_name = reporter.by_name(prof)
    for row in (".cablc", ".dswct", ".sinwp_sc", ".pn_isub", "array", "tmcsa", "control"):
        assert by_name.get(row, 0.0) > 0.0, f"missing/empty {row} row; have {sorted(by_name)}"
    for silent in ("cell", "adc"):
        assert silent not in by_name, f"unexpected self-billing module row {silent}: {sorted(by_name)}"


def test_static_report_seats_reporters_only(device: torch.device) -> None:
    macro = build_macro(build_config(), device=device)
    static = {e.qualified_name: e.leakage__uW for e in Reporter(macro).static_entries}
    for seat in ("", "control", "tmcsa_iref", "cablc", "sl_driver", "tmcsa"):
        assert seat in static, f"missing static seat {seat!r}; have {sorted(static)}"
        assert static[seat] > 0.0, f"non-positive leakage seat {seat!r}: {static[seat]}"
    assert "adc" not in static
    assert "array" in static, f"missing static seat 'array'; have {sorted(static)}"
    assert static["array"] == 0.0
    assert "pn_isub" not in static
    assert "dswct" not in static
    assert "sinwp_sc" not in static


def test_dynamic_energy_scales_with_conduction_windows(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(2)
    base_cfg = build_config(t_sample__ns=1.0, t_settle__ns=2.0)

    prof_base, rep_base = _run(base_cfg, w, x, device=device)
    dyn_base = rep_base.total_dynamic_energy__fJ(prof_base)
    assert dyn_base > 0.0

    prof_win, rep_win = _run(dataclasses.replace(base_cfg, t_settle__ns=4.0), w, x, device=device)
    assert rep_win.total_dynamic_energy__fJ(prof_win) > dyn_base  # dynamic grows with the window
    ch_base = _channels(rep_base.by_name(prof_base))
    ch_win = _channels(rep_win.by_name(prof_win))
    assert ch_win["control"] == pytest.approx(ch_base["control"])
    for ch in _READ_ROWS:
        assert ch_win[ch] > ch_base[ch], f"read channel {ch} did not grow with t_settle"


def test_runtime_adc_width_selects_every_sensing_window(device: torch.device) -> None:
    config = build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)
    w = _w_full()
    x = _x_full(3)
    rows: list[dict[str, float]] = []
    for bits in range(1, config.tmcsa_config.bits + 1):
        profiler, reporter = _run(config, w, x, device=device, adc_active_bits=bits)
        rows.append(reporter.by_name(profiler))

    for key in (".cablc", ".dswct", ".sinwp_sc", ".pn_isub", "tmcsa"):
        values = [row[key] for row in rows]
        assert values[0] < values[1] < values[2], f"{key} did not follow runtime ADC width: {values}"
    assert rows[0]["control"] == pytest.approx(rows[-1]["control"])


def test_input_branch_billed_whole_by_cablc_array_bills_caps_only(device: torch.device) -> None:
    cfg = build_config()
    w = _w_full()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)  # batch (2,)

    def run(config: Xue2020JsscCimMacroConfig) -> tuple[float, float, float]:
        macro = build_macro(config, device=device)
        macro.program(w.to(device))
        with Profiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
        by_name = Reporter(macro).by_name(prof)
        cablc = by_name.get(".cablc", 0.0)
        array = by_name.get("array", 0.0)
        assert "cell" not in by_name
        whole = _whole_input_branch(macro, x.to(device))
        return cablc, array, whole

    cablc, array, whole = run(cfg)
    # Keep the ownership assertions non-vacuous.
    assert whole > 0.0, f"witness draws no branch current: whole={whole}"
    assert cablc == pytest.approx(whole), f"cablc {cablc} != reconstructed whole branch {whole}"
    assert array > 0.0, f"array row {array} must bill its capacitive cycling"
    assert cablc + array > whole, f"array + cablc {cablc + array} must exceed the whole branch {whole} by the caps"

    cablc_wide, array_wide, whole_wide = run(dataclasses.replace(cfg, t_settle__ns=cfg.t_settle__ns + 3.0))
    assert array_wide == pytest.approx(array), (
        f"array row moved with the conduction window {array} -> {array_wide} (conduction leaked back into the array)"
    )
    assert cablc_wide == pytest.approx(whole_wide), f"cablc {cablc_wide} != reconstructed whole branch {whole_wide}"
    assert cablc_wide > cablc, f"cablc must grow with the conduction window {cablc} -> {cablc_wide}"


def test_array_cap_row_rides_the_shared_core_supply(device: torch.device) -> None:
    base = build_config()
    w = _w_full()
    x = _x_full(2)

    def array_row(config: Xue2020JsscCimMacroConfig) -> float:
        prof, reporter = _run(config, w, x, device=device)
        return reporter.by_name(prof)["array"]

    e_base = array_row(base)
    raised = array_row(dataclasses.replace(base, vdd__V=2.0 * base.vdd__V))
    assert e_base > 0.0
    assert raised == pytest.approx(2.0 * e_base)


def test_control_count_mux_times_batch(device: torch.device) -> None:
    cfg = build_config()
    e_per_op = cfg.control_config.energy_per_op__fJ
    w = _w_full()

    e1 = _channel_energies(cfg, w, torch.tensor([1, 2, 1, 0], dtype=torch.long), device=device)["control"]
    assert e1 == pytest.approx(e_per_op * cfg.scan_num)

    x_batch = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long)
    e_batch = _channel_energies(cfg, w, x_batch, device=device)["control"]
    assert e_batch == pytest.approx(e_per_op * cfg.scan_num * 3)


def test_pn_isub_channel_present_and_uses_detection(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(3)

    def pnisub(cfg: Xue2020JsscCimMacroConfig) -> float:
        return _channel_energies(cfg, w, x, device=device)["pn_isub"]

    base = pnisub(build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0))
    more_settle = pnisub(build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=3.0))
    more_sample = pnisub(build_config(x_bit_num=3, t_sample__ns=7.0, t_settle__ns=1.0))
    assert base > 0.0
    assert more_settle > base, "pn_isub must grow with detection"
    assert more_sample == pytest.approx(base), "pn_isub must be invariant to the sampled-bit windows"


def test_pn_isub_per_op_energy_is_billed_per_scan_and_lane(device: torch.device) -> None:
    config = build_config()
    event__fJ = 3.0
    without_event = dataclasses.replace(config, pn_isub_energy_per_op__fJ=0.0)
    with_event = dataclasses.replace(config, pn_isub_energy_per_op__fJ=event__fJ)
    w = _w_full()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long)

    e_without__fJ = _channel_energies(without_event, w, x, device=device)["pn_isub"]
    e_with__fJ = _channel_energies(with_event, w, x, device=device)["pn_isub"]
    event_num = x.shape[0] * config.scan_num * config.lane_num
    assert e_with__fJ - e_without__fJ == pytest.approx(event__fJ * event_num)


def test_read_channels_linear_in_t_sample(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(3)
    energies = [
        _channel_energies(
            build_config(x_bit_num=3, t_sample__ns=ts, t_settle__ns=1.0),
            w,
            x,
            device=device,
        )
        for ts in (2.0, 4.0, 6.0)
    ]
    for ch in _READ_ROWS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_sample: {v}"
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]), f"control not window-invariant: {ctrl}"
    assert ctrl[1] == pytest.approx(ctrl[2]), f"control not window-invariant: {ctrl}"


def test_read_channels_linear_in_t_settle(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(3)
    energies = [
        _channel_energies(
            build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=ts),
            w,
            x,
            device=device,
        )
        for ts in (1.0, 3.0, 5.0)  # equal spacing dt = 2
    ]
    for ch in _READ_ROWS:
        v = [e[ch] for e in energies]
        assert v[0] < v[1] < v[2], f"{ch} not increasing in t_settle: {v}"
        assert (v[1] - v[0]) == pytest.approx(v[2] - v[1], rel=1e-9, abs=1e-9), f"{ch} non-linear: {v}"
    ctrl = [e["control"] for e in energies]
    assert ctrl[0] == pytest.approx(ctrl[1]), f"control not window-invariant: {ctrl}"
    assert ctrl[1] == pytest.approx(ctrl[2]), f"control not window-invariant: {ctrl}"


def test_sc_earlier_current_stays_active_through_later_phases(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(3)
    base_cfg = build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)
    wide_cfg = dataclasses.replace(base_cfg, t_sample__ns=4.0)
    base = _channel_energies(base_cfg, w, x, device=device)["sinwp_sc"]
    wide = _channel_energies(wide_cfg, w, x, device=device)["sinwp_sc"]
    assert wide > base


def test_read_channel_per_bit_window_is_diagonal_not_suffix(device: torch.device) -> None:
    w = _w_full(input_num=TINY_INPUT_NUM)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.long)  # bit 0 set, bits 1..K-1 zero

    def read(t_sample: float, t_settle: float) -> dict[str, float]:
        return _channel_energies(
            build_config(x_bit_num=3, t_sample__ns=t_sample, t_settle__ns=t_settle),
            w,
            x,
            device=device,
        )

    base = read(2.0, 1.0)
    bump_input = read(4.0, 1.0)
    bump_settle = read(2.0, 3.0)

    for ch in ("cablc", "dswct"):
        assert bump_input[ch] > base[ch]
        assert bump_settle[ch] == pytest.approx(base[ch]), (
            f"{ch} bit-0 conduction leaked into detection: {base[ch]} -> {bump_settle[ch]}"
        )
    assert bump_settle["sinwp_sc"] > base["sinwp_sc"]


def test_live_bit_conducts_during_detection_independent_of_sampling(device: torch.device) -> None:
    cfg = build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)
    macro = build_macro(cfg, device=device)
    detect__ns = cfg.t_settle__ns + macro.tmcsa.latency__ns(active_bits=TINY_ADC_BITS)
    assert detect__ns == cfg.t_settle__ns + TINY_ADC_BITS * cfg.tmcsa_config.latency_per_bit__ns

    w = _w_full()
    x = _x_full(3)

    def cablc_settle_slope(t_sample: float) -> float:
        lo = _channel_energies(
            build_config(x_bit_num=3, t_sample__ns=t_sample, t_settle__ns=1.0),
            w,
            x,
            device=device,
        )["cablc"]
        hi = _channel_energies(
            build_config(x_bit_num=3, t_sample__ns=t_sample, t_settle__ns=3.0),
            w,
            x,
            device=device,
        )["cablc"]
        return (hi - lo) / 2.0

    slope_a = cablc_settle_slope(2.0)
    slope_b = cablc_settle_slope(7.0)
    assert slope_a > 0.0, "the live bit must conduct during detection"
    assert slope_a == pytest.approx(slope_b, rel=1e-9, abs=1e-9), (
        f"live-bit detection leaked into sampling: {(slope_a, slope_b)}"
    )


def test_tmcsa_per_bit_energy_remains_when_phase_windows_are_zero(device: torch.device) -> None:
    base = build_config()
    cfg = _with_tmcsa(base, t_ph2__ns=0.0, t_ph3__ns=0.0)
    cfg_no_bit_energy = _with_tmcsa(cfg, energy_per_bit__fJ=0.0)
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)  # single access (no batch axis)
    prof, reporter = _run(cfg, w, x, device=device)
    energy__fJ = reporter.by_name(prof)["tmcsa"]
    prof_no_bit_energy, reporter_no_bit_energy = _run(cfg_no_bit_energy, w, x, device=device)

    assert energy__fJ > 0.0
    assert reporter_no_bit_energy.by_name(prof_no_bit_energy)["tmcsa"] == pytest.approx(0.0)


def test_tmcsa_grows_with_phase_windows(device: torch.device) -> None:
    w = _w_full()
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    base = build_config()  # witness phases: t_ph2 = 0.2 ns, t_ph3 = 0.3 ns

    def tmcsa(cfg: Xue2020JsscCimMacroConfig) -> float:
        prof, reporter = _run(cfg, w, x, device=device)
        return reporter.by_name(prof)["tmcsa"]

    e_zero = tmcsa(_with_tmcsa(base, t_ph2__ns=0.0, t_ph3__ns=0.0))
    e_base = tmcsa(base)
    e_double = tmcsa(
        _with_tmcsa(
            base,
            t_ph2__ns=2.0 * base.tmcsa_config.t_ph2__ns,
            t_ph3__ns=2.0 * base.tmcsa_config.t_ph3__ns,
        )
    )

    assert e_base > e_zero
    assert e_double == pytest.approx(2.0 * e_base - e_zero)
