"""Xue2020 energy ownership and timing tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from neurox import Profiler
from tests.works.macro.cim.xue2020jssc.macro._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacroConfig,
    build_config,
    build_macro,
)

_READ_ROWS = ("cablc", "dswct", "sinwp_sc")  # sampled-phase conduction rows


def _w_full(input_num: int = TINY_INPUT_NUM, output_num: int = TINY_OUTPUT_NUM) -> Tensor:
    """All-`+1` weights so every physical column and scan conducts."""
    return torch.ones((input_num, output_num), dtype=torch.int32)


def _x_full(k: int, input_num: int = TINY_INPUT_NUM) -> Tensor:
    """Full-scale K-bit input (every input bit set on every row) — all sub-phase legs conduct."""
    return torch.full((input_num,), (1 << k) - 1, dtype=torch.int32)


def _energies(
    config: Xue2020JsscCimMacroConfig,
    w: Tensor,
    x: Tensor,
    *,
    device: torch.device,
    quantization_mode: int = QUANTIZATION_MODE,
    adc_active_bits: int = TINY_ADC_BITS,
) -> dict[str, float]:
    macro = build_macro(config, device=device)
    macro.program(w.to(device))
    with Profiler(concat_dim=0) as prof, torch.no_grad():
        macro.vec_mat_mul(
            x.to(device),
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
    return {
        name: item.dynamic_energy__fJ.sum().item()
        for name, item in prof.result.items()
        if item.dynamic_energy__fJ is not None
    }


def test_detection_and_sampling_keep_separate_energy_windows(device: torch.device) -> None:
    w = _w_full()
    x = _x_full(3)
    config = build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)

    def energies(t_sample: float, t_settle: float) -> dict[str, float]:
        return _energies(dataclasses.replace(config, t_sample__ns=t_sample, t_settle__ns=t_settle), w, x, device=device)

    base = energies(2.0, 1.0)
    detection = energies(2.0, 3.0)
    sampling = energies(7.0, 1.0)
    both = energies(7.0, 3.0)
    assert sum(detection.values()) > sum(base.values()) > 0.0
    # Conduction grows with its window; capacitive cycling and control do not.
    assert base["array"] > 0.0
    assert detection["array"] == pytest.approx(base["array"])
    assert detection["control"] == pytest.approx(base["control"])
    for channel in _READ_ROWS:
        assert detection[channel] > base[channel]
    assert detection["pn_isub"] > base["pn_isub"] > 0.0
    assert sampling["pn_isub"] == pytest.approx(base["pn_isub"])
    assert detection["cablc"] - base["cablc"] == pytest.approx(both["cablc"] - sampling["cablc"], rel=1e-9, abs=1e-9)


def test_runtime_adc_width_selects_every_sensing_window(device: torch.device) -> None:
    config = build_config(x_bit_num=3, t_sample__ns=2.0, t_settle__ns=1.0)
    w = _w_full()
    x = _x_full(3)
    rows = [
        _energies(config, w, x, device=device, adc_active_bits=bits) for bits in range(1, config.tmcsa_config.bits + 1)
    ]

    for key in ("cablc", "dswct", "sinwp_sc", "pn_isub", "tmcsa"):
        values = [row[key] for row in rows]
        assert values[0] < values[1] < values[2], f"{key} did not follow runtime ADC width: {values}"
    assert rows[0]["control"] == pytest.approx(rows[-1]["control"])


def test_pn_isub_per_op_energy_is_billed_per_scan_and_lane(device: torch.device) -> None:
    config = build_config()
    event__fJ = 3.0
    without_event = dataclasses.replace(config, pn_isub_energy_per_op__fJ=0.0)
    with_event = dataclasses.replace(config, pn_isub_energy_per_op__fJ=event__fJ)
    w = _w_full()
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.int32)

    e_without__fJ = _energies(without_event, w, x, device=device)["pn_isub"]
    e_with__fJ = _energies(with_event, w, x, device=device)["pn_isub"]
    event_num = x.shape[0] * config.scan_num * config.lane_num
    assert e_with__fJ - e_without__fJ == pytest.approx(event__fJ * event_num)


def test_read_channel_per_bit_window_is_diagonal_not_suffix(device: torch.device) -> None:
    w = _w_full(input_num=TINY_INPUT_NUM)
    x = torch.ones(TINY_INPUT_NUM, dtype=torch.int32)  # bit 0 set, bits 1..K-1 zero

    def read(t_sample: float, t_settle: float) -> dict[str, float]:
        return _energies(
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


def test_partial_outputs_close_columns_without_erasing_solved_currents(device: torch.device) -> None:
    config = build_config()
    macro = build_macro(config, device=device, inst_shape=(2,))
    macro.program(_w_full().to(device).expand(2, -1, -1))
    x = torch.ones((5, 1, TINY_INPUT_NUM), dtype=torch.int32, device=device)
    counts = torch.arange(5, device=device).unsqueeze(-1)
    x[0] = 0
    full = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=TINY_ADC_BITS)
    macro.set_profile_leading_rank(1)
    with Profiler(concat_dim=0) as profiler:
        partial = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=TINY_ADC_BITS, effective_output_num=counts)
    assert torch.equal(partial, full.where(torch.arange(TINY_OUTPUT_NUM, device=device) < counts.unsqueeze(-1), 0))
    for item in profiler.result.values():
        if item.dynamic_energy__fJ is not None:
            assert item.dynamic_energy__fJ[0] == 0
    duration__ns = macro.latency__ns(adc_active_bits=TINY_ADC_BITS, effective_output_num=counts)
    assert duration__ns[0] == 0
    assert duration__ns[1] == duration__ns[2]
    assert duration__ns[2] * 2 == duration__ns[3] == duration__ns[4]
    # Two macros each execute one control event per active scan. A second
    # valid output fills the other lane without adding a scan event.
    control__fJ = profiler.result["control"].dynamic_energy__fJ
    torch.testing.assert_close(
        control__fJ,
        control__fJ.new_tensor([0.0, 2.0, 2.0, 4.0, 4.0]) * config.control_config.energy_per_op__fJ,
        check_dtype=False,
    )
