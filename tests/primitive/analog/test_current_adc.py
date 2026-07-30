"""Single-ended current ADC: per-instance references + bits + B-form energy.

``SarIadc.convert(i_in__uA, i_refs__uA, *, bits)`` takes the reference ladder
per call as a ``[..., n_ref]`` tensor of ``2 ** max_bits - 1`` ascending taps on
the **last** axis (the caller has already selected the operating mode's row —
mode is invisible to the ADC); the leading dims broadcast right-aligned
against ``i_in__uA``, so each ADC instance may carry its own ladder. Bit width
is ADC-internal: the FULL ladder is always wired and a ``bits``-bit conversion
truncates the max-bits binary search after ``bits`` levels. These tests pin:

- per-call reference validation: the last axis must carry ``2 ** max_bits - 1``
  taps whatever ``bits`` is requested (any leading rank is accepted);
- per-instance broadcast: distinct ladders across the leading digitize their own
  inputs;
- ``bits`` validation: a request outside ``[1, max_bits]`` is rejected;
- config-time energy-knob validation: negative rail / window and a window /
  step-latency list shorter than ``bits``;
- conversion correctness: unit-step ladder codes = the count of taps the input
  exceeds; the truncated search starts at the max-bits mid tap and its code at
  ``b`` is the max-bits code right-shifted by ``max_bits - b``;
- B-form energy: fixed-only when the window (or rail) is zero — which also pins
  the base ``_compute_input_dynamic_energy__fJ`` hook at zero — and linear in both
  ``v_rail__V`` and ``t_conduct_per_step__ns``; energy and latency count the
  EXECUTED steps, so both scale with ``bits`` over the same full ladder;
  ``enable_latency_record=False`` suppresses the latency event while keeping the
  dynamic-energy event, and its mirror ``enable_energy_record=False`` suppresses
  the dynamic-energy event while keeping latency and the exact codes;
- ``Iadc.convert`` template method: probe-off equivalence with
  ``_convert_impl`` and :class:`IadcProber` capture of input,
  code, and resolution.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_adc import (
    IadcProber,
    SarIadc,
    SarIadcConfig,
    SarIadcPolicy,
)

# A 3-bit ladder (7 taps) with unit steps: code = count of taps the input exceeds.
_LADDER_A = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def _config(
    *,
    adc_bits: int = 3,
    v_rail__V: float = 0.0,
    t_conduct_per_step__ns: tuple[float, ...] = (0.0, 0.0, 0.0),
    step_latency__ns: tuple[float, ...] = (3.0, 3.0, 3.0),
    e_fixed_per_op__fJ: float = 7.0,
) -> SarIadcConfig:
    return SarIadcConfig(
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
    )


def _build(
    config: SarIadcConfig,
    device: torch.device,
    *,
    enable_latency_record: bool = True,
    enable_energy_record: bool = True,
) -> SarIadc:
    adc = SarIadc(
        config=config,
        policy=SarIadcPolicy(
            comparator_offset=False,
            coupling_mismatch=False,
        ),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
        enable_latency_record=enable_latency_record,
        enable_energy_record=enable_energy_record,
    )
    adc.to(device)
    adc.eval()
    adc.fabricate()
    return adc


def _refs(taps: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Per-call reference ladder with the ``2 ** bits - 1`` taps on the last axis."""
    return torch.tensor(taps, dtype=torch.float64, device=device)


def _convert_energy(adc: SarIadc, i_in: torch.Tensor, refs: torch.Tensor, adc_bits: int) -> float:
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


def test_convert_requires_the_full_ladder_at_every_bits(device: torch.device) -> None:
    """The tap count is the ADC's OWN capability, not the requested resolution."""
    adc = _build(_config(adc_bits=3), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for adc_bits in (1, 2, 3):
        # Short ladder: max_bits = 3 requires exactly 2**3 - 1 = 7 taps.
        with pytest.raises(ValueError, match="n_taps"):
            adc.convert(i_in, _refs(_LADDER_A[:6], device), bits=adc_bits)
        # The full ladder is accepted at every width.
        adc.convert(i_in, _refs(_LADDER_A, device), bits=adc_bits)
    # A ladder sized for the REQUESTED width is rejected below the maximum.
    for adc_bits in (1, 2):
        with pytest.raises(ValueError, match="n_taps"):
            adc.convert(i_in, _refs(_LADDER_A[: (1 << adc_bits) - 1], device), bits=adc_bits)


def test_convert_rejects_bad_bits(device: torch.device) -> None:
    """``bits`` outside ``[1, max_bits]`` is rejected."""
    adc = _build(_config(adc_bits=3), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for adc_bits in (0, -1, 4):
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
    """Distinct ladders across the leading digitize their own inputs."""
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


@pytest.mark.parametrize("max_bits", [1, 2, 4])
def test_lowered_bits_equal_the_max_bits_code_shifted(device: torch.device, max_bits: int) -> None:
    """Equivalence law: ``convert(bits=b) == convert(bits=B) >> (B - b)``.

    The whole ladder is wired at every width; a ``b``-bit conversion is the
    first ``b`` levels of the max-bits search tree, so it resolves exactly the
    max-bits code's leading ``b`` bits. Edge widths ``b = 1`` and ``b = B`` are
    covered by the sweep.
    """
    steps = (0.0,) * max_bits
    adc = _build(
        _config(adc_bits=max_bits, t_conduct_per_step__ns=steps, step_latency__ns=steps),
        device,
    )
    ladder = tuple(float(k + 1) for k in range((1 << max_bits) - 1))
    refs = _refs(ladder, device)
    # Sweep every bin, both saturation tails, and every exact tap value (threshold ties).
    i_in = torch.arange(-0.5, (1 << max_bits) + 0.5, 0.25, dtype=torch.float64, device=device)

    full = adc.convert(i_in, refs, bits=max_bits)
    assert int(full.max()) == (1 << max_bits) - 1
    for adc_bits in range(1, max_bits + 1):
        code = adc.convert(i_in, refs, bits=adc_bits)
        assert adc.unsigned_range(adc_bits) == (0, (1 << adc_bits) - 1)
        assert torch.equal(code, full >> (max_bits - adc_bits))


def test_first_compare_is_the_max_bits_mid_tap(device: torch.device) -> None:
    """A 1-bit conversion splits at the FULL ladder's midpoint, not at its own.

    With ``max_bits = 3`` the single decision sits at tap ``2**2 - 1`` (value
    4.0 on the unit ladder), so the code flips there and nowhere else.
    """
    adc = _build(_config(adc_bits=3), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 3.5, 4.5, 7.5], dtype=torch.float64, device=device)
    code = adc.convert(i_in, refs, bits=1)
    assert torch.equal(code, torch.tensor([0, 0, 1, 1], device=device))


# ---------------------------------------------------------------------------
# B-form per-step energy + latency
# ---------------------------------------------------------------------------


def test_energy_is_fixed_only_without_conduction(device: torch.device) -> None:
    """Zero window ⇒ pure ``bits * e_fixed`` model; the base hook adds nothing.

    With ``v_rail__V`` on but the window zero, the conduction term vanishes, so
    the whole per-conversion energy is exactly ``numel * bits * e_fixed`` — a
    surviving contribution from the ``_compute_input_dynamic_energy__fJ`` hook (base
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


def test_latency_sums_only_the_executed_step_windows(device: torch.device) -> None:
    """Latency at ``bits`` sums the first ``bits`` ``step_latency__ns`` entries.

    The ladder is the same full one at both widths — only the executed step
    count differs.
    """
    adc = _build(_config(adc_bits=3, step_latency__ns=(3.0, 5.0, 7.0)), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    refs = _refs(_LADDER_A, device)

    with NeuroxProfiler() as p3:
        adc.convert(i_in, refs, bits=3)
    with NeuroxProfiler() as p2:
        adc.convert(i_in, refs, bits=2)

    assert p3.total_latency__ns == pytest.approx(3.0 + 5.0 + 7.0)
    assert p2.total_latency__ns == pytest.approx(3.0 + 5.0)


def test_energy_counts_only_the_executed_steps(device: torch.device) -> None:
    """Fixed energy at ``bits`` is ``bits * e_fixed`` over the same full ladder."""
    adc = _build(_config(adc_bits=3, v_rail__V=1.0, e_fixed_per_op__fJ=7.0), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5], dtype=torch.float64, device=device)

    for adc_bits in (1, 2, 3):
        energy = _convert_energy(adc, i_in, refs, adc_bits=adc_bits)
        assert energy == pytest.approx(i_in.numel() * adc_bits * 7.0)


def test_enable_latency_record_false_suppresses_only_latency(device: torch.device) -> None:
    """``enable_latency_record=False`` drops the latency event but keeps the dynamic-energy event."""
    config = _config(adc_bits=3, step_latency__ns=(3.0, 5.0, 7.0), e_fixed_per_op__fJ=7.0)
    adc = _build(config, device, enable_latency_record=False)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    with NeuroxProfiler() as p:
        adc.convert(i_in, _refs(_LADDER_A, device), bits=3)

    assert p.total_latency__ns == pytest.approx(0.0)
    # Dynamic energy is unconditional (zero window ⇒ pure fixed floor).
    assert p.total_dynamic_energy__fJ == pytest.approx(i_in.numel() * 3 * 7.0)


def test_enable_energy_record_false_suppresses_only_energy(device: torch.device) -> None:
    """``enable_energy_record=False`` drops the dynamic-energy event but keeps latency and codes.

    The composition-boundary switch mirrors ``enable_latency_record``: an owner
    that bills conversion energy itself builds the ADC energy-silent, and the
    value conversion is untouched.
    """
    config = _config(
        adc_bits=3,
        v_rail__V=1.0,
        t_conduct_per_step__ns=(0.5, 0.5, 0.5),
        step_latency__ns=(3.0, 5.0, 7.0),
        e_fixed_per_op__fJ=7.0,
    )
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)
    refs = _refs(_LADDER_A, device)

    recording = _build(config, device)
    silent = _build(config, device, enable_energy_record=False)

    with NeuroxProfiler() as p_rec:
        code_rec = recording.convert(i_in, refs, bits=3)
    with NeuroxProfiler() as p_silent:
        code_silent = silent.convert(i_in, refs, bits=3)

    # Value conversion untouched; latency event kept; NO energy event at all.
    assert torch.equal(code_silent, code_rec)
    assert p_silent.total_latency__ns == pytest.approx(p_rec.total_latency__ns)
    assert p_rec.total_dynamic_energy__fJ > 0.0
    assert p_silent.total_dynamic_energy__fJ == pytest.approx(0.0)
    assert len(p_silent.energy_events) == 0


# ---------------------------------------------------------------------------
# Family template method (probe side channel)
# ---------------------------------------------------------------------------


def test_probe_preserves_output_and_captures_call(device: torch.device) -> None:
    """An active prober preserves conversion and records the call."""
    adc = _build(_config(), device)
    refs = _refs(_LADDER_A, device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)

    expected = adc.convert(i_in, refs, bits=3)
    with IadcProber() as prober:
        code = adc.convert(i_in, refs, bits=3)

    assert torch.equal(code, expected)
    records = prober.records
    assert len(records) == 1
    observation = records[0]
    assert torch.equal(observation.i_in__uA, i_in)
    assert torch.equal(observation.code, code)
    assert observation.bits == 3
