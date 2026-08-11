"""Tests for the ADC family.

Covers the two concrete implementations (:class:`GeneralDiffVadc`,
:class:`McsSarDiffVadc`) under the **raw-code output convention**: every
ADC's ``convert`` returns raw unsigned codes in ``[0, 2**bits - 1]``; the
zero point (``zero_offset`` / ``zero_code``) is subtracted consumer-side,
not inside the ADC. Both converters take their reference taps per call —
neither holds a reference value of its own — and each states its own tap
count, since the base validates only ``bits``.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog.diff_voltage_adc import (
    GeneralDiffVadc,
    GeneralDiffVadcConfig,
    GeneralDiffVadcPolicy,
    McsSarDiffVadc,
    McsSarDiffVadcConfig,
    McsSarDiffVadcPolicy,
)
from neurox.primitive.analog.voltage_reference import (
    Vref,
    VrefConfig,
    VrefPolicy,
)

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# GeneralDiffVadc convert — raw unsigned output
# ---------------------------------------------------------------------------


# 4-bit witness: 15 comparator thresholds -> 16 codes -> raw range [0, 15].
_GENERAL_CODE_NUM = 16


def _general_taps(*, scale: float = 1.0) -> torch.Tensor:
    """Ascending differential threshold ladder, symmetric around 0.

    The taps are differential comparison thresholds, so they run negative
    and no single-ended :class:`Vref` can source them; the owner builds the
    ladder and injects it per call.
    """
    return torch.tensor([(k - 7.5) * 0.1 * scale for k in range(_GENERAL_CODE_NUM - 1)], dtype=_DTYPE)


def _build_general_adc() -> GeneralDiffVadc:
    config = GeneralDiffVadcConfig(
        code_num=_GENERAL_CODE_NUM,
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        energy_per_op__fJ=0.0,
        latency_per_op__ns=1.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    policy = GeneralDiffVadcPolicy(sampling_noise=False, comparator_noise=False)
    adc = GeneralDiffVadc(
        config=config,
        policy=policy,
        inst_shape=(1,),
        dtype=_DTYPE,
        T__K=300.0,
    )
    adc.eval()
    return adc


class TestGeneralAdcRawConvert:
    def test_output_lies_in_raw_range(self) -> None:
        adc = _build_general_adc()

        v_pos = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(), bits=4)

        assert code.dtype == torch.int16
        # Raw unsigned bucket index, no zero shift applied by the ADC.
        assert adc.unsigned_range(4) == (0, 15)
        assert int(code.min()) >= 0
        assert int(code.max()) <= 15

    def test_midpoint_signal_maps_to_zero_code(self) -> None:
        # Ladder centered around 0 -> signal at 0 should land at the raw
        # zero code (bucket midpoint), so (code - zero_code) is ~0.
        adc = _build_general_adc()

        v_pos = torch.zeros(3, dtype=_DTYPE)
        v_neg = torch.zeros(3, dtype=_DTYPE)
        code = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(), bits=4)
        # The ADC does NOT fold the zero point in; the consumer subtracts
        # zero_code. Signal 0 lands on (or one bucket off) the midpoint.
        assert adc.zero_code == 8
        assert int((code - adc.zero_code).abs().max()) <= 1

    def test_signal_above_range_saturates_to_raw_max(self) -> None:
        adc = _build_general_adc()

        v_pos = torch.tensor([10.0], dtype=_DTYPE)  # way above range
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(), bits=4)
        # max raw code = 15 for 16 buckets
        assert int(code.item()) == adc.unsigned_range(4)[1] == 15

    def test_signal_below_range_saturates_to_raw_min(self) -> None:
        adc = _build_general_adc()

        v_pos = torch.tensor([-10.0], dtype=_DTYPE)  # way below range
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(), bits=4)
        # min raw code = 0
        assert int(code.item()) == adc.unsigned_range(4)[0] == 0

    def test_transfer_follows_the_injected_ladder(self) -> None:
        """The thresholds are the caller's, not the ADC's: widening them lowers the code."""
        adc = _build_general_adc()

        v_pos = torch.tensor([0.5], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        tight = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(), bits=4)
        wide = adc.convert(v_pos, v_neg, v_refs__V=_general_taps(scale=10.0), bits=4)
        assert int(tight.item()) > int(wide.item())


# ---------------------------------------------------------------------------
# McsSarDiffVadc convert — raw unsigned output, multi-mode, flexible bits
# ---------------------------------------------------------------------------


# One full-scale reference per mode row: the CDAC is single-tap, and a mode
# picks WHICH single tap, so the bank is [mode][tap] with tap_num == 1.
_MCS_MODE_REFS__V = (0.8, 0.4, 0.2)


def _mode_ref__V(mode: int) -> torch.Tensor:
    """Source one mode row's single full-scale tap from a per-mode Vref bank."""
    ref = Vref(
        config=VrefConfig(
            v_refs__V=tuple((v,) for v in _MCS_MODE_REFS__V),
            tolerance_sigma_relative=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VrefPolicy(tolerance=False),
        inst_shape=(),
        dtype=_DTYPE,
        T__K=300.0,
    )
    ref.fabricate()
    return ref.v_out__V[mode, :]


def _build_mcs_sar_adc(max_bits: int = 4) -> McsSarDiffVadc:
    config = McsSarDiffVadcConfig(
        max_bits=max_bits,
        clk_period__ns=2.0,
        c_unit__fF=2.0,
        cap_mismatch_sigma_relative=0.0,
        comparator_offset_sigma__V=0.0,
        comparator_thermal_noise_sigma__V=0.0,
        e_bootstrap__fJ=0.0,
        e_constant_per_bit__fJ=0.0,
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
    def test_output_lies_in_raw_range_at_max_bits(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0, -0.5, -1.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), bits=4)

        # Raw offset-binary code in [0, 15]; zero point subtracted consumer-side.
        assert adc.unsigned_range(4) == (0, 15)
        assert adc.zero_offset(4) == 8
        assert int(code.min()) >= 0
        assert int(code.max()) <= 15

    def test_output_lies_in_raw_range_at_lower_bits(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        # 2-bit: raw range [0, 3]
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), bits=2)
        assert adc.unsigned_range(2) == (0, 3)
        assert int(code.min()) >= 0
        assert int(code.max()) <= 3

    def test_extreme_positive_saturates_to_raw_max(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)

        v_pos = torch.tensor([100.0], dtype=_DTYPE)  # well beyond V_ref
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), bits=4)
        # Saturation at top: raw = 15 for 4 bits (all bits 1)
        assert int(code.item()) == 15

    def test_extreme_negative_saturates_to_raw_min(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)

        v_pos = torch.tensor([-100.0], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), bits=4)
        # Saturation at bottom: raw = 0 for 4 bits (all bits 0)
        assert int(code.item()) == 0

    def test_different_modes_select_different_reference_rows(self) -> None:
        # Mode 0 sources V_ref = 0.8, mode 2 sources V_ref = 0.2. A 0.5 V
        # signal saturates under the tight mode and stays linear under the
        # wide one. The ADC never sees the mode index: the source resolves
        # the row and hands over a single-tap bank.
        adc = _build_mcs_sar_adc(max_bits=4)

        v_pos = torch.tensor([0.5], dtype=_DTYPE)
        v_neg = torch.zeros_like(v_pos)
        code_mode0 = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(0), bits=4)
        code_mode2 = adc.convert(v_pos, v_neg, v_refs__V=_mode_ref__V(2), bits=4)
        assert int(code_mode2.item()) == 15  # saturated positive -> raw max
        assert int(code_mode0.item()) < 15  # within linear region


# ---------------------------------------------------------------------------
# Validation: each leaf states its own call contract
# ---------------------------------------------------------------------------


class TestConvertCallValidation:
    def test_mcs_sar_rejects_bits_out_of_range(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)

        v = torch.zeros(1, dtype=_DTYPE)
        with pytest.raises(ValueError, match="bits"):
            adc.convert(v, v, v_refs__V=_mode_ref__V(0), bits=5)

    def test_mcs_sar_rejects_a_multi_tap_bank(self) -> None:
        """The CDAC swings against ONE full-scale reference — its own circuit fact."""
        adc = _build_mcs_sar_adc(max_bits=4)

        v = torch.zeros(1, dtype=_DTYPE)
        with pytest.raises(ValueError, match="tap_num"):
            adc.convert(v, v, v_refs__V=torch.tensor([0.8, 0.4], dtype=_DTYPE), bits=4)

    def test_general_rejects_a_ladder_of_the_wrong_length(self) -> None:
        """``code_num`` comparators need exactly ``code_num - 1`` thresholds."""
        adc = _build_general_adc()

        v = torch.zeros(1, dtype=_DTYPE)
        with pytest.raises(ValueError, match="tap_num"):
            adc.convert(v, v, v_refs__V=_general_taps()[:-1], bits=4)
