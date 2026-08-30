"""Differential voltage ADC raw-code and call-contract tests.

`convert` returns raw offset-binary codes in `[0, 2**active_bits - 1]`;
the consumer subtracts `zero_offset(active_bits)`. References arrive per call,
while the concrete circuit owns its required tap count.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog import Reference, ReferenceConfig, ReferencePolicy
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
        T__K=300.0,
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
        T__K=300.0,
    )
    adc.eval()
    adc.fabricate()
    return adc


class TestMcsSarDiffVadcRawConvert:
    def test_output_lies_in_raw_range_at_full_width(self) -> None:
        adc = _build_mcs_sar_adc(bits=4)

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0, -0.5, -1.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), active_bits=4)

        # Raw offset-binary code in [0, 15]; zero point subtracted consumer-side.
        assert adc.unsigned_range(4) == (0, 15)
        assert adc.zero_offset(4) == 8
        assert int(code.min()) >= 0
        assert int(code.max()) <= 15

    def test_output_lies_in_raw_range_at_lower_bits(self) -> None:
        adc = _build_mcs_sar_adc(bits=4)

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        # 2-bit: raw range [0, 3]
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), active_bits=2)
        assert adc.unsigned_range(2) == (0, 3)
        assert int(code.min()) >= 0
        assert int(code.max()) <= 3

    def test_extreme_positive_saturates_to_raw_max(self) -> None:
        adc = _build_mcs_sar_adc(bits=4)

        v_pos = torch.tensor([100.0], dtype=_DTYPE)  # well beyond V_ref
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), active_bits=4)
        # Saturation at top: raw = 15 for 4 bits (all bits 1)
        assert int(code.item()) == 15

    def test_extreme_negative_saturates_to_raw_min(self) -> None:
        adc = _build_mcs_sar_adc(bits=4)

        v_pos = torch.tensor([-100.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), active_bits=4)
        # Saturation at bottom: raw = 0 for 4 bits (all bits 0)
        assert int(code.item()) == 0

    def test_different_modes_select_different_reference_rows(self) -> None:
        # Mode 0 sources V_ref = 0.8, mode 2 sources V_ref = 0.2. A 0.5 V
        # signal saturates under the tight mode and stays linear under the
        # wide one.
        adc = _build_mcs_sar_adc(bits=4)

        v_pos = torch.tensor([0.5], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code_mode0 = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), active_bits=4)
        code_mode2 = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(2), active_bits=4)
        assert int(code_mode2.item()) == 15  # saturated positive -> raw max
        assert int(code_mode0.item()) < 15  # within linear region


class TestConvertCallValidation:
    def test_mcs_sar_rejects_bits_out_of_range(self) -> None:
        adc = _build_mcs_sar_adc(bits=4)

        v = torch.zeros(1, dtype=_DTYPE)
        with pytest.raises(ValueError, match="bits"):
            adc.convert(v, v, v_refs__V=_mode_ref__V(0), active_bits=5)

    def test_mcs_sar_rejects_a_multi_tap_bank(self) -> None:
        """The CDAC swings against ONE full-scale reference — its own circuit fact."""
        adc = _build_mcs_sar_adc(bits=4)

        v = torch.zeros(1, dtype=_DTYPE)
        with pytest.raises(ValueError, match="tap_num"):
            adc.convert(v, v, v_refs__V=torch.tensor([0.8, 0.4], dtype=_DTYPE), active_bits=4)
