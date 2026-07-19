"""Voltage-ADC family template method: probe-off equivalence + probe capture.

``VoltageAdc.convert`` delegates to ``_convert_impl`` and emits the call on
the ``adc.convert`` probe channel. Without an active prober the template must
be bit-identical to the leaf conversion body; with one, the record must carry
the call's inputs, code, and operating-point fields.
"""

from __future__ import annotations

import torch

from neurox.common.prober import AdcProber, Prober
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.analog.voltage_adc import (
    GeneralVoltageAdc,
    GeneralVoltageAdcConfig,
    GeneralVoltageAdcPolicy,
)


def _build_general_adc(device: torch.device) -> GeneralVoltageAdc:
    # 4-bit: 15 boundaries -> 16 codes -> signed range [-8, 7]
    boundaries = tuple((k - 7.5) * 0.1 for k in range(15))
    config = GeneralVoltageAdcConfig(
        boundaries=boundaries,
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        input_transform="linear",
        energy_per_op__fJ=0.0,
        latency_per_op__ns=1.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = GeneralVoltageAdc(
        config=config,
        policy=GeneralVoltageAdcPolicy(sampling_noise=False, comparator_noise=False),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.to(device)
    adc.eval()
    return adc


# GeneralVoltageAdc is reference-free; a 1-tap dummy satisfies the signature.
def _dummy_vrefs(device: torch.device) -> torch.Tensor:
    return torch.zeros(1, dtype=torch.float64, device=device)


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_pos = torch.tensor([-1.0, -0.31, 0.0, 0.29, 1.0], dtype=torch.float64, device=device)
    return v_pos, torch.zeros_like(v_pos)


def test_probe_off_convert_matches_convert_impl(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)
    op = AdcOperationPoint(adc_mode=0, adc_bits=4)

    assert not Prober._active_stack
    via_template = adc.convert(v_pos, v_neg, v_refs__V=_dummy_vrefs(device), adc_operation_point=op)
    direct = adc._convert_impl(v_pos, v_neg, v_refs__V=_dummy_vrefs(device), adc_operation_point=op)
    assert torch.equal(via_template, direct)


def test_probe_capture_carries_inputs_code_and_op_point(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)
    v_refs = _dummy_vrefs(device)
    op = AdcOperationPoint(adc_mode=0, adc_bits=4)

    with AdcProber() as prober:
        out = adc.convert(v_pos, v_neg, v_refs__V=v_refs, adc_operation_point=op)

    records = prober.convert_records()
    assert len(records) == 1
    module, tensors = records[0]
    assert module is adc
    assert torch.equal(tensors["v_pos__V"], v_pos)
    assert torch.equal(tensors["v_neg__V"], v_neg)
    assert torch.equal(tensors["v_refs__V"], v_refs)
    assert torch.equal(tensors["code"], out)
    assert tensors["adc_mode"].item() == 0
    assert tensors["adc_bits"].item() == 4


def test_no_record_without_prober(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)
    op = AdcOperationPoint(adc_mode=0, adc_bits=4)

    with AdcProber() as outer:
        pass  # closed before the call: nothing may be recorded
    adc.convert(v_pos, v_neg, v_refs__V=_dummy_vrefs(device), adc_operation_point=op)
    assert outer.convert_records() == []
