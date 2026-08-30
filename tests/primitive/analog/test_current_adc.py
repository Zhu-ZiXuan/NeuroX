"""Single-ended current ADC: per-instance references and truncated SAR search.

`SarIadc.convert(i_in__uA, i_refs__uA, *, active_bits)` takes the reference ladder
per call as a `[..., n_ref]` tensor of `2 ** bits - 1` ascending taps on
the **last** axis (the caller has already selected the operating mode's row —
mode is invisible to the ADC); the leading dims broadcast right-aligned
against `i_in__uA`, so each ADC instance may carry its own ladder. Bit width
is ADC-internal: the FULL ladder is always wired and an `active_bits` conversion
truncates the configured binary search after that many levels. These tests pin:

- per-instance broadcast: distinct ladders across the leading digitize their own
  inputs (any leading rank is accepted);
- `active_bits` validation: a request outside `[1, bits]` is rejected;
- config-time timing validation: negative step latency;
- conversion correctness: unit-step ladder codes = the count of taps the input
  meets or exceeds; the truncated search starts at the full-width mid tap and
  its code at `b` is the full-width code right-shifted by `bits - b`; positive
  comparator offset raises the reference-side threshold;
- `Iadc.convert` template method: probe-off equivalence with
  `_convert_impl` and `AdcProber` capture of the input signal.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog import AdcProber
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
        T__K=300.0,
    )
    adc.to(device)
    adc.eval()
    adc.fabricate()
    return adc


def _refs(taps: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Per-call reference ladder with the `2 ** bits - 1` taps on the last axis."""
    return torch.tensor(taps, dtype=torch.float64, device=device)


# ---------------------------------------------------------------------------
# Config-time + convert-time validation
# ---------------------------------------------------------------------------


def test_config_rejects_negative_bit_latency() -> None:
    with pytest.raises(ValueError, match=r"require: latency_per_bit__ns \(-0\.1\) >= 0"):
        _config(latency_per_bit__ns=-0.1)


def test_convert_rejects_bad_active_bits(device: torch.device) -> None:
    """`active_bits` outside `[1, bits]` is rejected."""
    adc = _build(_config(bits=3), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for active_bits in (0, -1, 4):
        with pytest.raises(ValueError, match="active_bits"):
            adc.convert(i_in, _refs(_LADDER_A, device), active_bits=active_bits)


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


def test_unit_ladder_code_counts_reached_taps(device: torch.device) -> None:
    """The unit-step ladder counts the taps the input meets or exceeds."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0, 100.0], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, active_bits=3)
    assert torch.equal(code, torch.tensor([0, 4, 7, 7], device=device))


def test_per_instance_ladders_broadcast(device: torch.device) -> None:
    """Distinct ladders across the leading digitize their own inputs."""
    adc = _build(_config(), device)
    # Two instances, each with its own 7-tap ladder on the last axis.
    refs = torch.tensor(
        [_LADDER_A, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)],
        dtype=torch.float64,
        device=device,
    )
    i_in = torch.tensor([4.5, 25.0], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, active_bits=3)
    # Row 0 clears 4 unit taps; row 1 clears the 10 and 20 taps.
    assert torch.equal(code, torch.tensor([4, 2], device=device))


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
    ladder = tuple(float(k + 1) for k in range((1 << bits) - 1))
    refs = _refs(ladder, device)
    # Sweep every bin, both saturation tails, and every exact tap value (threshold ties).
    i_in = torch.arange(-0.5, (1 << bits) + 0.5, 0.25, dtype=torch.float64, device=device)

    full = adc.convert(i_in, refs, active_bits=bits)
    assert int(full.max()) == (1 << bits) - 1
    for active_bits in range(1, bits + 1):
        code = adc.convert(i_in, refs, active_bits=active_bits)
        assert adc.unsigned_range(active_bits) == (0, (1 << active_bits) - 1)
        assert torch.equal(code, full >> (bits - active_bits))


def test_first_compare_is_the_full_width_mid_tap(device: torch.device) -> None:
    """A 1-bit conversion splits at the FULL ladder's midpoint, not at its own.

    With `bits = 3` the single decision sits at tap `2**2 - 1` (value
    4.0 on the unit ladder), so the code flips there and nowhere else.
    """
    adc = _build(_config(bits=3), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 3.5, 4.0, 4.5, 7.5], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, active_bits=1)
    assert torch.equal(code, torch.tensor([0, 0, 1, 1, 1], device=device))


def test_positive_comparator_offset_raises_reference_threshold(device: torch.device) -> None:
    """Positive input-referred offset is added to the negative reference port."""
    adc = _build(_config(bits=3), device)
    adc._comparator_offset__uA = torch.full(adc.inst_shape, 0.25, dtype=torch.float64, device=device)

    i_in = torch.tensor([4.0, 4.24, 4.25], dtype=torch.float64, device=device)
    code = adc.convert(i_in, _refs(_LADDER_A, device), active_bits=1)

    assert torch.equal(code, torch.tensor([0, 0, 1], device=device))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_reported_latency_scales_with_executed_steps(device: torch.device) -> None:
    """`latency__ns` is the fixed step period times the executed bit count."""
    adc = _build(_config(bits=3, latency_per_bit__ns=3.0), device)

    assert adc.latency__ns(active_bits=3) == pytest.approx(9.0)
    assert adc.latency__ns(active_bits=2) == pytest.approx(6.0)
    assert adc.latency__ns(active_bits=1) == pytest.approx(3.0)
    # No request outside the physical resolution.
    for active_bits in (0, 4):
        with pytest.raises(ValueError, match=r"require: active_bits \(\d+\) in \[1, bits \(3\)\]"):
            adc.latency__ns(active_bits=active_bits)


# ---------------------------------------------------------------------------
# Family template method (probe side channel)
# ---------------------------------------------------------------------------


def test_probe_preserves_output_and_captures_call(device: torch.device) -> None:
    """An active prober preserves conversion and records the call."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    expected = adc.convert(i_in, refs, active_bits=3)
    # The record stays where it was recorded, which is where the call's own
    # tensors it is compared against live.
    with AdcProber() as prober:
        code = adc.convert(i_in, refs, active_bits=3)

    assert torch.equal(code, expected)
    records = prober.records
    assert len(records) == 1
    record = records[0]
    assert torch.equal(record.i_in__uA, i_in)
    assert record.input_name() == "i_in__uA"
    assert torch.equal(record.input_value(), i_in)
    assert not hasattr(record, "code")
    assert not hasattr(record, "bits")
