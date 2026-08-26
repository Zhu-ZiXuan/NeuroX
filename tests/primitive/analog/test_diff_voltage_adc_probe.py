"""Voltage-ADC family template method: probe-off equivalence + probe capture.

`DiffVadc.convert` delegates to `_convert_impl` and emits its input on the
shared `AdcProber`. Without an active prober the template remains bit-identical
to the leaf conversion body; with one, the record carries only the input
terminals. The injected reference is calibrated design data, not a measured
quantity, so it is not recorded.
"""

from __future__ import annotations

import torch

from neurox.primitive.analog import AdcProber
from neurox.primitive.analog.diff_voltage_adc import (
    GeneralDiffVadc,
    GeneralDiffVadcConfig,
    GeneralDiffVadcPolicy,
)

# 4-bit: 15 comparator thresholds -> 16 codes -> raw range [0, 15]
_CODE_NUM = 16


def _build_general_adc(device: torch.device) -> GeneralDiffVadc:
    config = GeneralDiffVadcConfig(
        code_num=_CODE_NUM,
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        energy_per_op__fJ=0.0,
        latency_per_op__ns=1.0,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    adc = GeneralDiffVadc(
        config=config,
        policy=GeneralDiffVadcPolicy(sampling_noise=False, comparator_noise=False),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.to(device)
    adc.eval()
    return adc


def _taps(device: torch.device) -> torch.Tensor:
    """The injected comparator ladder, owner-built and passed in per call."""
    return torch.tensor([(k - 7.5) * 0.1 for k in range(_CODE_NUM - 1)], dtype=torch.float64, device=device)


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    v_pos = torch.tensor([-1.0, -0.31, 0.0, 0.29, 1.0], dtype=torch.float64, device=device)
    return v_pos, torch.zeros_like(v_pos)


def test_probe_preserves_output_and_captures_call(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)
    v_refs = _taps(device)

    expected = adc.convert(v_pos, v_neg, v_refs__V=v_refs, bits=4)
    # The record stays where it was recorded, which is where the call's own
    # tensors it is compared against live.
    with AdcProber() as prober:
        out = adc.convert(v_pos, v_neg, v_refs__V=v_refs, bits=4)

    assert torch.equal(out, expected)
    records = prober.records
    assert len(records) == 1
    record = records[0]
    assert torch.equal(record.v_pos__V, v_pos)
    assert torch.equal(record.v_neg__V, v_neg)
    assert record.input_name() == "v_diff__V"
    assert torch.equal(record.input_value(), v_pos - v_neg)
    assert not hasattr(record, "code")
    assert not hasattr(record, "bits")
    # A reference is calibrated design data, not part of the conversion event.
    assert not hasattr(record, "v_ref__V")
    assert not hasattr(record, "v_refs__V")


def test_no_record_without_prober(device: torch.device) -> None:
    adc = _build_general_adc(device)
    v_pos, v_neg = _inputs(device)

    with AdcProber() as outer:
        pass  # closed before the call: nothing may be recorded
    adc.convert(v_pos, v_neg, v_refs__V=_taps(device), bits=4)
    assert outer.records == ()
