"""Differential voltage ADC energy recording and temperature-dependent noise."""

from __future__ import annotations

import torch

from neurox import Profiler, stamp_names
from neurox.primitive.analog import AdcProber, Reference, ReferenceConfig, ReferencePolicy
from neurox.primitive.analog.diff_voltage_adc import (
    McsSarDiffVadc,
    McsSarDiffVadcConfig,
    McsSarDiffVadcPolicy,
)

_DTYPE = torch.float64


_MCS_MODE_REFS__V = (0.8, 0.4, 0.2)


def _mode_ref__V(mode: int) -> torch.Tensor:
    ref = Reference(
        config=ReferenceConfig(
            values=_MCS_MODE_REFS__V,
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=ReferencePolicy(tolerance=False),
        inst_shape=(),
        dtype=_DTYPE,
    )
    ref.fabricate()
    return ref.values()[mode].unsqueeze(-1)


def _build_mcs_sar_adc(bits: int = 4) -> McsSarDiffVadc:
    config = McsSarDiffVadcConfig(
        bits=bits,
        latency_per_bit__ns=2.0,
        c_unit__fF=2.0,
        cap_mismatch_sigma_relative=0.0,
        comparator_offset_sigma__V=0.0,
        comparator_thermal_noise_sigma__V=0.0,
        energy_per_op__fJ=0.0,
        energy_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    policy = McsSarDiffVadcPolicy(
        cap_mismatch=False,
        comparator_offset=False,
        comparator_thermal_noise=False,
        sampling_thermal_noise=False,
    )
    adc = McsSarDiffVadc(
        config=config,
        policy=policy,
        inst_shape=(1,),
        dtype=_DTYPE,
    )
    adc.eval()
    adc.fabricate()
    return adc


def test_public_conversion_records_returned_energy_once() -> None:
    adc = _build_mcs_sar_adc()
    stamp_names(adc)
    v_pos = torch.tensor([0.12, 0.3, 0.6], dtype=_DTYPE)
    v_neg = torch.zeros_like(v_pos)
    refs = _mode_ref__V(0)

    with torch.no_grad():
        expected_code, expected_energy = adc._convert_impl(
            v_pos, v_neg, v_refs__V=refs, active_bits=4, record_energy=True
        )
    without_records = adc.convert(v_pos, v_neg, v_refs__V=refs, active_bits=4)
    with Profiler(leading_rank=1) as profiler, AdcProber() as prober:
        code = adc.convert(v_pos, v_neg, v_refs__V=refs, active_bits=4)

    torch.testing.assert_close(code, expected_code)
    torch.testing.assert_close(without_records, expected_code)
    assert len(profiler.records) == 1
    assert len(prober.records) == 1
    torch.testing.assert_close(profiler.records[0].dynamic_energy__fJ, expected_energy)


def test_thermal_noise_scales_with_temperature(device: torch.device) -> None:
    adc = McsSarDiffVadc(
        config=McsSarDiffVadcConfig(
            bits=3,
            latency_per_bit__ns=1.0,
            c_unit__fF=2.0,
            cap_mismatch_sigma_relative=0.1,
            comparator_offset_sigma__V=0.01,
            comparator_thermal_noise_sigma__V=0.02,
            energy_per_op__fJ=0.0,
            energy_per_bit__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=McsSarDiffVadcPolicy(
            cap_mismatch=True,
            comparator_offset=False,
            comparator_thermal_noise=True,
            sampling_thermal_noise=True,
        ),
        inst_shape=(4,),
        dtype=torch.float64,
    ).to(device)
    adc.fabricate()
    v_pos = torch.linspace(-0.2, 0.2, 512, dtype=torch.float64, device=device).reshape(128, 4)
    v_neg = torch.zeros_like(v_pos)
    v_refs = torch.tensor([0.2], dtype=torch.float64, device=device)
    torch.manual_seed(61)
    before = adc.convert(v_pos, v_neg, v_refs__V=v_refs, active_bits=3)
    adc.set_temperature(675.0)
    # Scaling signal and reference with sqrt(T) preserves signal-to-noise ratio.
    scale = 1.5
    torch.manual_seed(61)
    after = adc.convert(v_pos * scale, v_neg * scale, v_refs__V=v_refs * scale, active_bits=3)
    assert torch.equal(after, before)
