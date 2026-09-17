"""SAR ladder search, truncated decisions, and comparator-offset polarity."""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog.current_adc import (
    SarIadc,
    SarIadcConfig,
    SarIadcPolicy,
)

# A 3-bit ladder (7 taps): code = count of taps the input meets or exceeds.
_LADDER_A = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def _config(
    *,
    bits: int = 3,
    latency_per_bit__ns: float = 3.0,
) -> SarIadcConfig:
    return SarIadcConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        bits=bits,
        latency_per_bit__ns=latency_per_bit__ns,
        comparator_offset_sigma__uA=0.0,
    )


def _build(
    config: SarIadcConfig,
    device: torch.device,
) -> SarIadc:
    adc = SarIadc(
        config=config,
        policy=SarIadcPolicy(comparator_offset=False),
        inst_shape=(1,),
        dtype=torch.float64,
    )
    adc.to(device)
    adc.eval()
    adc.fabricate()
    return adc


def _refs(taps: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Per-call reference ladder with the `2 ** bits - 1` taps on the last axis."""
    return torch.tensor(taps, dtype=torch.float64, device=device)


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_lowered_bits_equal_the_full_width_code_shifted(device: torch.device, bits: int) -> None:
    """Equivalence law: `convert(active_bits=b) == full >> (B - b)`.

    The whole ladder is wired at every width; a `b`-bit conversion is the
    first `b` levels of the full-width search tree, so it resolves exactly the
    full-width code's leading `b` bits. Edge widths `b = 1` and `b = B` are
    covered by the sweep.
    """
    adc = _build(
        _config(bits=bits, latency_per_bit__ns=0.0),
        device,
    )
    ladder = torch.arange(1, 1 << bits, dtype=torch.float64, device=device)
    refs = torch.stack((ladder, ladder * 2.0))
    # Sweep every bin, both saturation tails, and every exact tap value (threshold ties).
    i_in = torch.arange(-0.5, 2 * (1 << bits) + 0.5, 0.25, dtype=torch.float64, device=device)
    i_in = i_in.unsqueeze(-1).expand(-1, 2)

    full = adc.convert(i_in, refs, active_bits=bits)
    # Count reached taps independently of the binary search, including exact ties.
    expected = (i_in.unsqueeze(-1) >= refs).sum(dim=-1)
    assert torch.equal(full, expected)
    for active_bits in range(1, bits + 1):
        code = adc.convert(i_in, refs, active_bits=active_bits)
        assert torch.equal(code, full >> (bits - active_bits))


def test_positive_comparator_offset_raises_reference_threshold(device: torch.device) -> None:
    """Positive input-referred offset is added to the negative reference port."""
    adc = _build(_config(bits=3), device)
    adc._comparator_offset__uA = torch.full(adc.inst_shape, 0.25, dtype=torch.float64, device=device)

    i_in = torch.tensor([4.0, 4.24, 4.25], dtype=torch.float64, device=device)
    code = adc.convert(i_in, _refs(_LADDER_A, device), active_bits=1)

    assert torch.equal(code, torch.tensor([0, 0, 1], device=device))


# ---------------------------------------------------------------------------
# Family template method (probe side channel)
# ---------------------------------------------------------------------------
