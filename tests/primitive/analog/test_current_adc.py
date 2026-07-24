"""Single-ended current ADC: per-instance references + bits + B-form energy.

``SarSingleEndedCurrentAdc.convert(i_in__uA, i_refs__uA, *, bits)`` takes the reference
ladder per call as a ``[*R, n_ref]`` tensor of ``2 ** bits - 1`` ascending taps
on the **last** axis (the caller has already selected the operating mode's row —
mode is invisible to the ADC); the ``[*R]`` leading broadcasts right-aligned
against ``i_in__uA``, so each ADC instance may carry its own ladder. The
resolution ``bits`` is passed directly. These tests pin:

- per-call reference validation: a wrong ``n_taps`` (``!= 2 ** bits - 1`` on the
  last axis) is rejected at convert time (any leading rank is now accepted);
- per-instance broadcast: distinct ladders across the leading digitize their own
  inputs;
- ``bits`` validation: a request outside ``[1, config.bits]`` is rejected;
- config-time energy-knob validation: negative rail / window and a window /
  step-latency list shorter than ``bits``;
- conversion correctness: unit-step ladder codes = the count of taps the input
  exceeds; a call at a lower ``bits`` (< the physical max) digitizes at that
  resolution;
- B-form energy: fixed-only when the window (or rail) is zero — which also pins
  the base ``_input_dynamic_energy__fJ`` hook at zero — and linear in both
  ``v_rail__V`` and ``t_conduct_per_step__ns``; latency sums only the first
  ``bits`` step windows, and ``record_latency=False`` suppresses the latency
  event while keeping the dynamic-energy event;
- ``SingleEndedCurrentAdc.convert`` template method: probe-off equivalence with
  ``_convert_impl`` and :class:`SingleEndedCurrentAdcProber` capture of input,
  code, and resolution.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_adc import (
    SarSingleEndedCurrentAdc,
    SarSingleEndedCurrentAdcConfig,
    SarSingleEndedCurrentAdcPolicy,
    SingleEndedCurrentAdcProber,
)

# A 3-bit ladder (7 taps) with unit steps: code = count of taps the input exceeds.
_LADDER_A = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
# A 2-bit ladder (3 taps) for the reduced-resolution call.
_LADDER_2B = (1.0, 2.0, 3.0)


def _config(
    *,
    adc_bits: int = 3,
    v_rail__V: float = 0.0,
    t_conduct_per_step__ns: tuple[float, ...] = (0.0, 0.0, 0.0),
    step_latency__ns: tuple[float, ...] = (3.0, 3.0, 3.0),
    e_fixed_per_op__fJ: float = 7.0,
) -> SarSingleEndedCurrentAdcConfig:
    return SarSingleEndedCurrentAdcConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        bits=adc_bits,
        margin_gain=3.0,
        e_fixed_per_op__fJ=e_fixed_per_op__fJ,
        v_rail__V=v_rail__V,
        t_conduct_per_step__ns=t_conduct_per_step__ns,
        step_latency__ns=step_latency__ns,
        comparator_offset_sigma__uA=0.0,
        coupling_mismatch_sigma__uA=0.0,
        mirror_mismatch_sigma_relative=0.0,
    )


def _build(
    config: SarSingleEndedCurrentAdcConfig,
    device: torch.device,
    *,
    record_latency: bool = True,
) -> SarSingleEndedCurrentAdc:
    adc = SarSingleEndedCurrentAdc(
        config=config,
        policy=SarSingleEndedCurrentAdcPolicy(
            comparator_offset=False,
            replica_threshold_variation=False,
            mirror_mismatch=False,
            coupling_mismatch=False,
        ),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
        record_latency=record_latency,
    )
    adc.to(device)
    adc.eval()
    adc.fabricate()
    return adc


def _refs(taps: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Per-call reference ladder with the ``2 ** bits - 1`` taps on the last axis."""
    return torch.tensor(taps, dtype=torch.float64, device=device)


def _convert_energy(adc: SarSingleEndedCurrentAdc, i_in: torch.Tensor, refs: torch.Tensor, adc_bits: int) -> float:
    with NeuroxProfiler() as profiler:
        adc.convert(i_in, refs, bits=adc_bits)
    return profiler.total_dynamic_energy__fJ


# ---------------------------------------------------------------------------
# Config-time + convert-time validation
# ---------------------------------------------------------------------------


def test_config_rejects_bad_energy_knobs() -> None:
    """Negative rail / window entry and too-short window / latency lists are rejected."""
    for bad in (
        {"v_rail__V": -0.1},
        {"t_conduct_per_step__ns": (0.1, -0.1, 0.1)},
        {"t_conduct_per_step__ns": (0.1, 0.1)},  # shorter than bits (3)
        {"step_latency__ns": (3.0, 3.0)},  # shorter than bits (3)
    ):
        with pytest.raises(ValueError):
            _config(**bad)
    # Lists longer than bits are tolerated (only the first bits are drawn).
    assert _config(t_conduct_per_step__ns=(0.1, 0.1, 0.1, 0.1), step_latency__ns=(3.0, 3.0, 3.0, 3.0)).bits == 3


def test_convert_rejects_wrong_tap_count(device: torch.device) -> None:
    """A wrong last-axis tap count is rejected at convert time (any leading rank is fine)."""
    adc = _build(_config(), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    # Wrong n_taps: bits = 3 requires exactly 2**3 - 1 = 7 taps on the last axis.
    with pytest.raises(ValueError, match="n_taps"):
        adc.convert(i_in, _refs(_LADDER_A[:6], device), bits=3)


def test_convert_rejects_bad_bits(device: torch.device) -> None:
    """``bits`` outside ``[1, config.bits]`` is rejected."""
    adc = _build(_config(adc_bits=3), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for adc_bits in (0, 4):
        with pytest.raises(ValueError, match="bits"):
            adc.convert(i_in, _refs(_LADDER_A, device), bits=adc_bits)


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


def test_unit_ladder_code_counts_exceeded_taps(device: torch.device) -> None:
    """The unit-step ladder yields code = number of taps the input exceeds."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0, 100.0], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, bits=3)
    assert torch.equal(code, torch.tensor([0, 4, 7, 7], device=device))


def test_per_instance_ladders_broadcast(device: torch.device) -> None:
    """Distinct ladders across the leading digitize their own inputs (``[*R, n_ref]``)."""
    adc = _build(_config(), device)
    # Two instances, each with its own 7-tap ladder on the last axis.
    refs = torch.tensor(
        [_LADDER_A, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)],
        dtype=torch.float64,
        device=device,
    )
    i_in = torch.tensor([4.5, 25.0], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, bits=3)
    # Row 0 (unit taps): 4.5 exceeds 4 taps. Row 1 (decade taps): 25 exceeds 10, 20.
    assert torch.equal(code, torch.tensor([4, 2], device=device))


def test_lower_bits_digitizes_at_that_resolution(device: torch.device) -> None:
    """A call at bits < config.bits runs a shorter search over a 3-tap ladder."""
    adc = _build(_config(adc_bits=3), device)
    refs = _refs(_LADDER_2B, device)  # 2**2 - 1 = 3 taps
    i_in = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, bits=2)
    # Unit-step 3-tap ladder → code = count of taps exceeded, capped at 2**2 - 1 = 3.
    assert torch.equal(code, torch.tensor([0, 1, 2, 3], device=device))
    assert adc.unsigned_range(2) == (0, 3)


# ---------------------------------------------------------------------------
# B-form per-step energy + latency
# ---------------------------------------------------------------------------


def test_energy_is_fixed_only_without_conduction(device: torch.device) -> None:
    """Zero window ⇒ pure ``bits * e_fixed`` model; the base hook adds nothing.

    With ``v_rail__V`` on but the window zero, the conduction term vanishes, so
    the whole per-conversion energy is exactly ``numel * bits * e_fixed`` — a
    surviving contribution from the ``_input_dynamic_energy__fJ`` hook (base
    zero) would break this equality.
    """
    adc = _build(_config(v_rail__V=1.0, t_conduct_per_step__ns=(0.0, 0.0, 0.0), e_fixed_per_op__fJ=7.0), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    energy = _convert_energy(adc, i_in, refs, adc_bits=3)
    assert energy == pytest.approx(i_in.numel() * 3 * 7.0)


def test_energy_linear_in_window_and_rail(device: torch.device) -> None:
    """Conduction energy above the fixed floor is linear in both window and rail."""
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    e_floor = _convert_energy(
        _build(_config(v_rail__V=0.0, t_conduct_per_step__ns=(1.0, 1.0, 1.0)), device), i_in, refs, adc_bits=3
    )
    e_t = _convert_energy(
        _build(_config(v_rail__V=1.0, t_conduct_per_step__ns=(1.0, 1.0, 1.0)), device), i_in, refs, adc_bits=3
    )
    e_2t = _convert_energy(
        _build(_config(v_rail__V=1.0, t_conduct_per_step__ns=(2.0, 2.0, 2.0)), device), i_in, refs, adc_bits=3
    )
    e_2v = _convert_energy(
        _build(_config(v_rail__V=2.0, t_conduct_per_step__ns=(1.0, 1.0, 1.0)), device), i_in, refs, adc_bits=3
    )

    conduction = e_t - e_floor
    assert conduction > 0.0
    assert (e_2t - e_floor) == pytest.approx(2.0 * conduction)  # linear in the window
    assert (e_2v - e_floor) == pytest.approx(2.0 * conduction)  # linear in the rail


def test_latency_sums_only_the_requested_step_windows(device: torch.device) -> None:
    """Latency at ``bits`` sums the first ``bits`` ``step_latency__ns`` entries."""
    adc = _build(_config(adc_bits=3, step_latency__ns=(3.0, 5.0, 7.0)), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)

    with NeuroxProfiler() as p3:
        adc.convert(i_in, _refs(_LADDER_A, device), bits=3)
    with NeuroxProfiler() as p2:
        adc.convert(i_in, _refs(_LADDER_2B, device), bits=2)

    assert p3.total_latency__ns == pytest.approx(3.0 + 5.0 + 7.0)
    assert p2.total_latency__ns == pytest.approx(3.0 + 5.0)


def test_record_latency_false_suppresses_only_latency(device: torch.device) -> None:
    """``record_latency=False`` drops the latency event but keeps the dynamic-energy event."""
    config = _config(adc_bits=3, step_latency__ns=(3.0, 5.0, 7.0), e_fixed_per_op__fJ=7.0)
    adc = _build(config, device, record_latency=False)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    with NeuroxProfiler() as p:
        adc.convert(i_in, _refs(_LADDER_A, device), bits=3)

    assert p.total_latency__ns == pytest.approx(0.0)
    # Dynamic energy is unconditional (zero window ⇒ pure fixed floor).
    assert p.total_dynamic_energy__fJ == pytest.approx(i_in.numel() * 3 * 7.0)


# ---------------------------------------------------------------------------
# Family template method (probe side channel)
# ---------------------------------------------------------------------------


def test_probe_off_convert_matches_convert_impl(device: torch.device) -> None:
    """Without an active prober the template equals the leaf conversion body."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    assert not SingleEndedCurrentAdcProber._active_stack
    via_template = adc.convert(i_in, refs, bits=3)
    direct = adc._convert_impl(i_in, refs, bits=3)
    assert torch.equal(via_template, direct)


def test_probe_capture_carries_input_code_and_bits(device: torch.device) -> None:
    """An active prober records the call's input, code, and resolution."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    with SingleEndedCurrentAdcProber() as prober:
        code = adc.convert(i_in, refs, bits=3)

    records = prober.records
    assert len(records) == 1
    observation = records[0]
    assert torch.equal(observation.i_in__uA, i_in)
    assert torch.equal(observation.code, code)
    assert observation.bits == 3
