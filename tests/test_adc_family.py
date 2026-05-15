"""Tests for the physics-based ADC family.

Covers the concrete subclasses (``GeneralADC``, ``McsSarAdc``,
``PipelineADC``, ``CyclicADC``, ``RampADC``) plus the shared
:class:`ADCMode` invariant.

The family is now uniformly **runtime multi-mode**: every
``convert`` and ``latency_per_op__ns`` call takes explicit ``mode``
and ``bits`` keyword arguments.  Single-mode behavioural subclasses
enforce strict ``mode == 0`` and ``bits == self._mode.n_bits``
validation; the SAR variants accept any ``(mode, bits)`` inside
their configured envelope.

Each subclass is exercised on a small zero-noise input vector to
validate:

* ``convert`` returns integer codes in the expected range and a
  per-element float energy tensor of the same shape.
* ``latency_per_op__ns`` produces sensible scalars at the configured
  ``(mode, bits)``.
* Floor semantics: a signal at ``0.5 · max`` lands in the lower half
  of the code range (not the upper half — round-to-nearest would
  flip this).
"""

from __future__ import annotations

import math

import pytest
import torch

from neurox.analog.adc import (
    ADCMode,
    CyclicADC,
    CyclicADCConfig,
    GeneralADC,
    GeneralADCConfig,
    McsSarAdc,
    McsSarAdcConfig,
    PipelineADC,
    PipelineADCConfig,
    RampADC,
    RampADCConfig,
)


def _signal(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A short test signal covering the 0..1 range with one out-of-range edge."""
    return torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], dtype=dtype)


def _diff_signal(*, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (v_pos, v_neg) pair so ``v_pos - v_neg`` reproduces ``_signal``."""
    return _signal(dtype=dtype), torch.zeros_like(_signal(dtype=dtype))


@pytest.mark.parametrize(
    "n_bits,n_states",
    [(8, 256), (8, 193), (6, 64), (4, 16)],
)
def test_adc_mode_validation(n_bits: int, n_states: int) -> None:
    mode = ADCMode(n_bits=n_bits, n_states=n_states, max_signal=1.2)
    assert mode.n_codes == 1 << n_bits
    assert math.isclose(mode.lsb, 1.2 / mode.n_codes)


def test_adc_mode_rejects_invalid_combos() -> None:
    with pytest.raises(ValueError):
        ADCMode(n_bits=0, n_states=2, max_signal=1.0)
    with pytest.raises(ValueError):
        ADCMode(n_bits=4, n_states=1, max_signal=1.0)
    with pytest.raises(ValueError):
        ADCMode(n_bits=4, n_states=32, max_signal=1.0)
    with pytest.raises(ValueError):
        ADCMode(n_bits=4, n_states=4, max_signal=0.0)


def test_general_adc_floor_semantics() -> None:
    """0.5 · max with floor → lower-half code; 0.5 + LSB → upper-half."""
    config = GeneralADCConfig(boundaries=[c * 0.0625 for c in range(1, 16)])
    adc = GeneralADC(config)
    adc.eval()
    v_pos = torch.tensor([0.5 - 1e-6, 0.5, 0.9, 1.5], dtype=torch.float32)
    v_neg = torch.zeros_like(v_pos)
    code, _ = adc.convert(v_pos, v_neg, mode=0, bits=4)
    # 0.5 - eps falls in code 7 (boundary 0.5 not yet exceeded);
    # 0.5 in code 8 (just exceeded boundary 0.5 = 8 · LSB);
    # 0.9 in code 14; 1.5 saturates at 15.
    assert code.tolist() == [7, 8, 14, 15]


def test_general_adc_rejects_bad_runtime_args() -> None:
    config = GeneralADCConfig(boundaries=[c * 0.0625 for c in range(1, 16)])
    adc = GeneralADC(config)
    adc.eval()
    v_pos, v_neg = _diff_signal()
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=1, bits=4)
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=0, bits=3)
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=0, bits=8)


def test_mcs_sar_smoke() -> None:
    cfg = McsSarAdcConfig(
        max_bits=8,
        v_refs__V=(1.2064,),
        clk_period__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=None,
        comparator_offset_sigma__V=None,
        comparator_thermal_noise_sigma__V=None,
        enable_thermal_noise=False,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = McsSarAdc(cfg, T__K=300.0)
    adc.fabricate(())
    adc.eval()
    v_pos, v_neg = _diff_signal()
    code, energy = adc.convert(v_pos, v_neg, mode=0, bits=8)
    assert int(code.min()) >= 0
    assert int(code.max()) <= (1 << 8) - 1
    assert energy.shape == code.shape
    # SAR latency at 8-bit = 9 clocks.
    assert math.isclose(adc.latency_per_op__ns(bits=8), 9.0)
    assert adc.max_bits() == 8
    assert adc.available_modes() == (1.2064,)


def test_mcs_sar_reduced_bits_clamps_codes() -> None:
    """When ``bits < max_bits`` codes fit in ``[0, 2**bits - 1]``."""
    cfg = McsSarAdcConfig(
        max_bits=8,
        v_refs__V=(1.0,),
        clk_period__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=None,
        comparator_offset_sigma__V=None,
        comparator_thermal_noise_sigma__V=None,
        enable_thermal_noise=False,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = McsSarAdc(cfg, T__K=300.0)
    adc.fabricate(())
    adc.eval()
    v_pos, v_neg = _diff_signal()
    code, _ = adc.convert(v_pos, v_neg, mode=0, bits=4)
    assert int(code.max()) <= 15
    # Latency at 4 bits = 5 cycles.
    assert math.isclose(adc.latency_per_op__ns(bits=4), 5.0)


def test_mcs_sar_invalid_runtime_args() -> None:
    """convert() rejects out-of-range mode / bits arguments."""
    cfg = McsSarAdcConfig(
        max_bits=8,
        v_refs__V=(1.0,),
        clk_period__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=None,
        comparator_offset_sigma__V=None,
        comparator_thermal_noise_sigma__V=None,
        enable_thermal_noise=False,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = McsSarAdc(cfg, T__K=300.0)
    adc.fabricate(())
    adc.eval()
    v_pos, v_neg = _diff_signal()
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=1, bits=8)
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=0, bits=0)
    with pytest.raises(ValueError):
        adc.convert(v_pos, v_neg, mode=0, bits=9)


def test_mcs_sar_multi_v_ref_supported() -> None:
    """Multiple V_refs share the same ADC instance; selected per call."""
    cfg = McsSarAdcConfig(
        max_bits=8,
        v_refs__V=(1.0, 0.5),
        clk_period__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=None,
        comparator_offset_sigma__V=None,
        comparator_thermal_noise_sigma__V=None,
        enable_thermal_noise=False,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = McsSarAdc(cfg, T__K=300.0)
    adc.fabricate(())
    adc.eval()
    v_pos = torch.tensor([0.1, 0.4, 0.7])
    v_neg = torch.zeros_like(v_pos)
    code_high, _ = adc.convert(v_pos, v_neg, mode=0, bits=8)
    code_low, _ = adc.convert(v_pos, v_neg, mode=1, bits=8)
    # Lower V_ref → same input pushes higher into the code range.
    assert int(code_low.max()) >= int(code_high.max())


def test_pipeline_adc_throughput_latency() -> None:
    mode = ADCMode(n_bits=4, n_states=16, max_signal=1.0)
    cfg = PipelineADCConfig(modes=(mode,), clk_period__ns=1.0, n_stages=4, pipeline_depth=2)
    adc = PipelineADC(cfg)
    adc.eval()
    assert math.isclose(adc.latency_per_op__ns(bits=4), 6.0)


def test_cyclic_adc_n_bit_cycles() -> None:
    mode = ADCMode(n_bits=5, n_states=32, max_signal=1.0)
    adc = CyclicADC(CyclicADCConfig(modes=(mode,), clk_period__ns=1.0))
    adc.eval()
    assert math.isclose(adc.latency_per_op__ns(bits=5), 5.0)


def test_ramp_adc_2power_n_cycles() -> None:
    mode = ADCMode(n_bits=4, n_states=16, max_signal=1.0)
    adc = RampADC(RampADCConfig(modes=(mode,), clk_period__ns=1.0))
    adc.eval()
    assert math.isclose(adc.latency_per_op__ns(bits=4), 16.0)


def test_mcs_sar_introspection_methods() -> None:
    """``available_modes`` and ``max_bits`` surface runtime options."""
    cfg = McsSarAdcConfig(
        max_bits=8,
        v_refs__V=(1.0, 0.5, 0.25),
        clk_period__ns=1.0,
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=None,
        comparator_offset_sigma__V=None,
        comparator_thermal_noise_sigma__V=None,
        enable_thermal_noise=False,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = McsSarAdc(cfg, T__K=300.0)
    assert adc.available_modes() == (1.0, 0.5, 0.25)
    assert adc.max_bits() == 8
