"""Voltage-ADC family template method: probe-off equivalence + probe capture.

``DifferentialVoltageAdc.convert`` delegates to ``_convert_impl`` and emits the
call on :class:`DifferentialVoltageAdcProber`. Without an active prober the
template must be bit-identical to the leaf conversion body; with one, the
record must carry the call's inputs, code, selected reference tap, and bits.
"""

from __future__ import annotations

import torch

from neurox.primitive.analog.voltage_adc import (
    DifferentialVoltageAdcProber,
    GeneralDifferentialVoltageAdc,
    GeneralDifferentialVoltageAdcConfig,
    GeneralDifferentialVoltageAdcPolicy,
)


def _build_general_adc(device: torch.device) -> GeneralDifferentialVoltageAdc:
    # 4-bit: 15 boundaries -> 16 codes -> raw range [0, 15]
    boundaries = tuple((k - 7.5) * 0.1 for k in range(15))
    config = GeneralDifferentialVoltageAdcConfig(
        boundaries=boundaries,
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        input_transform="linear",
        energy_per_op__fJ=0.0,
        latency_per_op__ns=1.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = GeneralDifferentialVoltageAdc(
        config=config,
        policy=GeneralDifferentialVoltageAdcPolicy(sampling_noise=False, comparator_noise=False),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.to(device)
    adc.eval()
    return adc


# GeneralDifferentialVoltageAdc is reference-free; a preselected dummy tap satisfies the signature.
def _dummy_vref(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float64, device=device)


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_pos = torch.tensor([-1.0, -0.31, 0.0, 0.29, 1.0], dtype=torch.float64, device=device)
    return v_pos, torch.zeros_like(v_pos)


def test_probe_preserves_output_and_captures_call(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)
    v_ref = _dummy_vref(device)

    expected = adc.convert(v_pos, v_neg, v_ref__V=v_ref, bits=4)
    with DifferentialVoltageAdcProber() as prober:
        out = adc.convert(v_pos, v_neg, v_ref__V=v_ref, bits=4)

    assert torch.equal(out, expected)
    records = prober.records
    assert len(records) == 1
    observation = records[0]
    assert torch.equal(observation.v_pos__V, v_pos)
    assert torch.equal(observation.v_neg__V, v_neg)
    assert torch.equal(observation.v_ref__V, v_ref)
    assert torch.equal(observation.code, out)
    assert observation.bits == 4


def test_no_record_without_prober(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)

    with DifferentialVoltageAdcProber() as outer:
        pass  # closed before the call: nothing may be recorded
    adc.convert(v_pos, v_neg, v_ref__V=_dummy_vref(device), bits=4)
    assert outer.records == []
