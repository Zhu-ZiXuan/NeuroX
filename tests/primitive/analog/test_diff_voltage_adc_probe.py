"""Voltage-ADC probing preserves conversion and captures input terminals."""

from __future__ import annotations

import torch

from neurox.primitive.analog import AdcProber
from neurox.primitive.analog.diff_voltage_adc import (
    McsSarDiffVadc,
    McsSarDiffVadcConfig,
    McsSarDiffVadcPolicy,
)


def _build_adc(device: torch.device) -> McsSarDiffVadc:
    config = McsSarDiffVadcConfig(
        bits=4,
        latency_per_bit__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=0.0,
        comparator_offset_sigma__V=0.0,
        comparator_thermal_noise_sigma__V=0.0,
        energy_per_op__fJ=0.0,
        energy_per_bit__fJ=0.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )
    adc = McsSarDiffVadc(
        config=config,
        policy=McsSarDiffVadcPolicy(
            cap_mismatch=False,
            comparator_offset=False,
            comparator_thermal_noise=False,
            sampling_thermal_noise=False,
        ),
        inst_shape=(1,),
        dtype=torch.float64,
    )
    adc.fabricate()
    adc.to(device)
    adc.eval()
    return adc


def _taps(device: torch.device) -> torch.Tensor:
    return torch.tensor([0.8], dtype=torch.float64, device=device)


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_pos = torch.tensor([-1.0, -0.31, 0.0, 0.29, 1.0], dtype=torch.float64, device=device)
    return v_pos, torch.zeros_like(v_pos)


def test_probe_preserves_output_and_captures_call(device: torch.device) -> None:
    adc = _build_adc(device)
    v_pos, v_neg = _inputs(device)
    v_refs = _taps(device)

    expected = adc.convert(v_pos, v_neg, v_refs__V=v_refs, active_bits=4)
    # The record stays where it was recorded, which is where the call's own
    # tensors it is compared against live.
    with AdcProber() as prober:
        out = adc.convert(v_pos, v_neg, v_refs__V=v_refs, active_bits=4)

    assert torch.equal(out, expected)
    records = prober.records
    assert len(records) == 1
    record = records[0]
    assert torch.equal(record.v_pos__V, v_pos)
    assert torch.equal(record.v_neg__V, v_neg)
    assert record.input_name() == "v_diff__V"
    assert torch.equal(record.input_value(), v_pos - v_neg)
