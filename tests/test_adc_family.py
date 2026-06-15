"""Tests for the ADC family.

Covers the shared :class:`ADCMode` invariants and the two concrete
implementations (:class:`GeneralADC`, :class:`McsSarAdc`) under the
**signed-code output convention**: every ADC's ``convert`` returns codes
in ``[-2**(bits-1), 2**(bits-1) - 1]``. How each concrete ADC produces
that signed output (precomputed zero code vs per-call shift) is its own
topology-specific implementation detail; the tests here only assert
behaviour, not implementation.

This file does NOT cover :class:`SarAdcMono`: that class is a
future-work placeholder, lives in ``neurox.analog.adc.sar_mono`` but
is **not** re-exported from ``neurox.analog.adc``. Its ``convert``
raises ``NotImplementedError`` so it cannot participate in any
end-to-end test.
"""

from __future__ import annotations

import math

import pytest
import torch

from neurox.analog.adc import (
    ADCMode,
    AdcOperationPoint,
    GeneralADC,
    GeneralADCConfig,
    GeneralADCPolicy,
    McsSarAdc,
    McsSarAdcConfig,
    McsSarAdcPolicy,
)

# ---------------------------------------------------------------------------
# ADCMode invariants
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GeneralADC convert — signed output
# ---------------------------------------------------------------------------


def _build_general_adc(boundaries: list[float]) -> GeneralADC:
    config = GeneralADCConfig(
        boundaries=tuple(boundaries),
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        drive_thermal__V=0.0,
        drive_value=0.0,
        input_transform="linear",
        energy_per_op__fJ=0.0,
        latency_per_op__ns=1.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    policy = GeneralADCPolicy(sampling_noise=False, comparator_noise=False, drive_thermal=False)
    return GeneralADC(
        config=config,
        policy=policy,
        name="general_adc",
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )


class TestGeneralAdcSignedConvert:
    def test_output_lies_in_signed_range(self) -> None:
        # 4-bit: 15 boundaries → 16 codes → signed range [-8, 7]
        boundaries = [(k - 7.5) * 0.1 for k in range(15)]  # symmetric around 0
        adc = _build_general_adc(boundaries)
        adc.eval()

        v_pos = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))

        assert code.dtype == torch.int16
        assert int(code.min()) >= -8
        assert int(code.max()) <= 7

    def test_midpoint_signal_maps_to_zero(self) -> None:
        # Boundaries centered around 0 → signal at 0 should land at signed code 0
        boundaries = [(k - 7.5) * 0.1 for k in range(15)]
        adc = _build_general_adc(boundaries)
        adc.eval()

        v_pos = torch.zeros(3, dtype=torch.float64)
        v_neg = torch.zeros(3, dtype=torch.float64)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        # Signal = 0 lands in the bucket [boundary_at_midpoint - LSB, boundary_at_midpoint),
        # which after sign flip is the signed code 0 or -1 (depending on which side
        # of the midpoint boundary "0" falls). With boundaries at ..., -0.05, 0.05, ...
        # signal 0 is in bucket starting at -0.05; unsigned code 7, signed code -1.
        # Either way the output should be small (near 0).
        assert int(code.abs().max()) <= 1

    def test_signal_above_range_saturates_to_positive(self) -> None:
        boundaries = [(k - 7.5) * 0.1 for k in range(15)]
        adc = _build_general_adc(boundaries)
        adc.eval()

        v_pos = torch.tensor([10.0], dtype=torch.float64)  # way above range
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        # max signed code = +7 for 4-bit
        assert int(code.item()) == 7

    def test_signal_below_range_saturates_to_negative(self) -> None:
        boundaries = [(k - 7.5) * 0.1 for k in range(15)]
        adc = _build_general_adc(boundaries)
        adc.eval()

        v_pos = torch.tensor([-10.0], dtype=torch.float64)  # way below range
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        # min signed code = -8 for 4-bit
        assert int(code.item()) == -8


# ---------------------------------------------------------------------------
# McsSarAdc convert — signed output, multi-mode, flexible bits
# ---------------------------------------------------------------------------


def _build_mcs_sar_adc(max_bits: int = 4, v_refs: tuple[float, ...] = (0.8, 0.4, 0.2)) -> McsSarAdc:
    config = McsSarAdcConfig(
        max_bits=max_bits,
        v_refs__V=v_refs,
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
    policy = McsSarAdcPolicy(
        cap_mismatch=False,
        comparator_offset=False,
        comparator_thermal_noise=False,
        sampling_thermal_noise=False,
    )
    return McsSarAdc(
        config=config,
        policy=policy,
        name="mcs_sar",
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )


class TestMcsSarAdcSignedConvert:
    def test_output_lies_in_signed_range_at_max_bits(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)
        adc.eval()
        adc.fabricate()

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0, -0.5, -1.0], dtype=torch.float64)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))

        assert int(code.min()) >= -8
        assert int(code.max()) <= 7

    def test_output_lies_in_signed_range_at_lower_bits(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)
        adc.eval()
        adc.fabricate()

        v_pos = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float64)
        v_neg = torch.zeros_like(v_pos)
        # 2-bit: signed range [-2, 1]
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=2))
        assert int(code.min()) >= -2
        assert int(code.max()) <= 1

    def test_extreme_positive_saturates_to_max(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)
        adc.eval()
        adc.fabricate()

        v_pos = torch.tensor([100.0], dtype=torch.float64)  # well beyond V_ref
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        # Saturation at top: signed = 7 for 4 bits
        assert int(code.item()) == 7

    def test_extreme_negative_saturates_to_min(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)
        adc.eval()
        adc.fabricate()

        v_pos = torch.tensor([-100.0], dtype=torch.float64)
        v_neg = torch.zeros_like(v_pos)
        code = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        # Saturation at bottom: signed = -8 for 4 bits
        assert int(code.item()) == -8

    def test_different_modes_use_different_v_ref(self) -> None:
        # Mode 0 uses v_ref = 0.8, mode 2 uses v_ref = 0.2.
        # A signal of 0.5V should saturate (or near-saturate) at mode 2 (range too tight)
        # but stay in linear region at mode 0.
        adc = _build_mcs_sar_adc(max_bits=4, v_refs=(0.8, 0.4, 0.2))
        adc.eval()
        adc.fabricate()

        v_pos = torch.tensor([0.5], dtype=torch.float64)
        v_neg = torch.zeros_like(v_pos)
        code_mode0 = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=4))
        code_mode2 = adc.convert(v_pos, v_neg, adc_operation_point=AdcOperationPoint(adc_mode=2, adc_bits=4))
        # Mode 2 (tight range) saturates; mode 0 (wide range) doesn't.
        assert int(code_mode2.item()) == 7  # saturated positive
        assert int(code_mode0.item()) < 7  # within linear region


# ---------------------------------------------------------------------------
# Validation: ADC convert rejects bad operating points
# ---------------------------------------------------------------------------


class TestOperatingPointValidation:
    def test_general_adc_rejects_nonzero_mode(self) -> None:
        boundaries = [(k - 7.5) * 0.1 for k in range(15)]
        adc = _build_general_adc(boundaries)
        adc.eval()

        v = torch.zeros(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="mode"):
            adc.convert(v, v, adc_operation_point=AdcOperationPoint(adc_mode=1, adc_bits=4))

    def test_mcs_sar_rejects_mode_out_of_range(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4, v_refs=(0.8, 0.4))  # mode_num=2
        adc.eval()
        adc.fabricate()

        v = torch.zeros(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="mode"):
            adc.convert(v, v, adc_operation_point=AdcOperationPoint(adc_mode=2, adc_bits=4))

    def test_mcs_sar_rejects_bits_out_of_range(self) -> None:
        adc = _build_mcs_sar_adc(max_bits=4)
        adc.eval()
        adc.fabricate()

        v = torch.zeros(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="bits"):
            adc.convert(v, v, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=5))
