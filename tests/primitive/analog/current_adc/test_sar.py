"""SAR ladder search, truncated decisions, and comparator-offset polarity."""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog.current_adc import (
    SarIadc,
    SarIadcConfig,
    SarIadcPolicy,
)


def _build(bits: int, device: torch.device) -> SarIadc:
    adc = SarIadc(
        config=SarIadcConfig(
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            bits=bits,
            latency_per_bit__ns=0.0,
            comparator_offset_sigma__uA=0.0,
        ),
        policy=SarIadcPolicy(comparator_offset=False),
        inst_shape=(1,),
        dtype=torch.float64,
    )
    adc.to(device)
    adc.eval()
    adc.fabricate()
    return adc


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_lowered_bits_equal_the_full_width_code_shifted(device: torch.device, bits: int) -> None:
    adc = _build(bits, device)
    ladder = torch.arange(1, 1 << bits, dtype=torch.float64, device=device)
    refs = torch.stack((ladder, ladder * 2.0))
    # Sweep every bin, both saturation tails, and every exact tap value (threshold ties).
    i_in = torch.arange(-0.5, 2 * (1 << bits) + 0.5, 0.25, dtype=torch.float64, device=device)
    i_in = i_in.unsqueeze(-1).expand(-1, 2)

    full = adc.convert(i_in, i_refs__uA=refs, active_bits=bits)
    # Count reached taps independently of the binary search, including exact ties.
    expected = (i_in.unsqueeze(-1) >= refs).sum(dim=-1)
    assert torch.equal(full, expected)
    for active_bits in range(1, bits):
        code = adc.convert(i_in, i_refs__uA=refs, active_bits=active_bits)
        assert torch.equal(code, full >> (bits - active_bits))


def test_positive_comparator_offset_raises_reference_threshold(device: torch.device) -> None:
    adc = _build(3, device)
    adc._comparator_offset__uA = torch.full(adc.inst_shape, 0.25, dtype=torch.float64, device=device)

    i_in = torch.tensor([4.0, 4.24, 4.25], dtype=torch.float64, device=device)
    refs = torch.arange(1, 8, dtype=torch.float64, device=device)
    code = adc.convert(i_in, i_refs__uA=refs, active_bits=1)

    assert torch.equal(code, torch.tensor([0, 0, 1], device=device))
