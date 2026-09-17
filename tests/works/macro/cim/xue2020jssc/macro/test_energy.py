"""Xue2020 energy ownership and timing tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from neurox import Profiler, Reporter
from tests.works.macro.cim.xue2020jssc.macro._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacroConfig,
    build_config,
    build_macro,
)

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
    return torch.ones((input_num, output_num), dtype=torch.int32)


def _x_full(k: int, input_num: int = TINY_INPUT_NUM) -> Tensor:
    """Full-scale K-bit input (every input bit set on every row) — all sub-phase legs conduct."""
    return torch.full((input_num,), (1 << k) - 1, dtype=torch.int32)


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
        macro.vec_mat_mul(
            x.to(device),
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
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


def test_channels_and_module_rows_bill_dynamic(device: torch.device) -> None:
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.int32)
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
    rows_base = rep_base.by_name(prof_base)
    rows_win = rep_win.by_name(prof_win)
    # The owner bills conduction; the array bills capacitive cycling once.
    assert rows_base["array"] > 0.0
    assert rows_win["array"] == pytest.approx(rows_base["array"])
    ch_base = _channels(rows_base)
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


def test_control_count_mux_times_batch(device: torch.device) -> None:
    cfg = build_config()
    e_per_op = cfg.control_config.energy_per_op__fJ
    w = _w_full()

    e1 = _channel_energies(cfg, w, torch.tensor([1, 2, 1, 0], dtype=torch.int32), device=device)["control"]
    assert e1 == pytest.approx(e_per_op * cfg.scan_num)

    x_batch = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.int32)
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
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.int32)

    e_without__fJ = _channel_energies(without_event, w, x, device=device)["pn_isub"]
    e_with__fJ = _channel_energies(with_event, w, x, device=device)["pn_isub"]
    event_num = x.shape[0] * config.scan_num * config.lane_num
    assert e_with__fJ - e_without__fJ == pytest.approx(event__fJ * event_num)


def test_read_channel_per_bit_window_is_diagonal_not_suffix(device: torch.device) -> None:
    w = _w_full(input_num=TINY_INPUT_NUM)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.int32)  # bit 0 set, bits 1..K-1 zero

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


def test_partial_outputs_close_columns_without_erasing_solved_currents(device: torch.device) -> None:
    macro = build_macro(build_config(), device=device, inst_shape=(2,))
    macro.program(_w_full().to(device).expand(2, -1, -1))
    x = torch.ones((5, 1, TINY_INPUT_NUM), dtype=torch.int32, device=device)
    counts = torch.arange(5, device=device).unsqueeze(-1)
    x[0] = 0
    full = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=TINY_ADC_BITS)
    with Profiler(leading_rank=1) as profiler:
        partial = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=TINY_ADC_BITS, effective_output_num=counts)
    assert torch.equal(partial, full.where(torch.arange(TINY_OUTPUT_NUM, device=device) < counts.unsqueeze(-1), 0))
    for record in profiler.records:
        assert record.dynamic_energy__fJ[0] == 0
    duration = macro.latency__ns(adc_active_bits=TINY_ADC_BITS, effective_output_num=counts)
    assert duration[0] == 0
    assert duration[1] * 2 == duration[2]
    assert duration[2] == duration[3] == duration[4]
