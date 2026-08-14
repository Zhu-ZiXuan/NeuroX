"""CPU-only eager tests for the ye2023jssc RS-CSA readout.

The Reference-Subtracting CSA is a UNIFORM current quantizer with an
OWNER-INJECTED static PH0 compensation, a DERIVED per-resolution conversion
window, and a per-compare-phase SAR energy. It takes ONE reference current and
scales it by its own compare-phase weights, so the decision ladder is internal.
Laws only (hand-written witness config; tiny shapes; eager, dynamo disabled):

  * the conversion window is DERIVED, not configured, and follows the EXECUTED
    phases: `t_conversion(b) == sum(t_phase[:b]) + t_intrinsic[b-1]` — PH0, the
    first `b - 1` compare phases in full, and the last executed compare phase
    up to ITS OWN comparator latch; it grows strictly with `b` and closes at
    the nominal `sum(t_phase[:-1]) + t_intrinsic[-1]` at full resolution,
  * PH0 is a construction-time constant (no config field): it shifts every
    conversion by the same current, and an input at or below it reads code 0,
  * the compare phases are a uniform quantizer whose step IS the injected
    reference: `code == floor((i_in - i_ph0)+ / i_ref)` clamped to the 4-bit
    ceiling,
  * the reference input is SINGLE-tap — the converter's own circuit fact, stated
    here and nowhere above it, so a deeper tap axis is rejected,
  * `config.bits` bounds the width a conversion may request; bit width is
    handled INSIDE the converter, which converts at full resolution and drops
    the code's low bits,
  * energy is `E_fixed(b) + E_code(b)` over the EXECUTED phases, with
    `E_code = sum_{p<=b} mirror_scale * v_rail * min(residue_p, 2**(B-p)*i_ref) * t_phase[p]`
    over the cumulative-subtraction residue (the latched REFS branches are
    rail-energy-neutral and are NOT billed) and the code-independent baseline
    prorated by the executed-window ratio, `E_fixed(b) = E_fixed * T_AC(b) /
    T_AC(B)`: a zero-residue conversion costs the prorated baseline alone, so
    its energy ratio between two resolutions IS the window ratio,
  * latency is the executed window — reported by `latency__ns` for the phase
    axis this converter owns,
  * conversion is deterministic — no jitter is wired, so `train()` and
    `eval()` return the same codes.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox import Profiler, Reporter, stamp_names
from neurox.works.macro.cim.ye2023jssc.rscsa import (
    RsCsaIadc,
    RsCsaIadcConfig,
    RsCsaIadcPolicy,
)

_BITS = 4
_I_REF__uA = 0.5  # the ONE injected reference; the code step is this current
_I_PH0__uA = 1.0  # = 2 * i_ref -> a clean 2-code static shift
_V_RAIL__V = 0.8
_T_PHASE__ns = (1.0, 2.0, 4.0, 8.0, 16.0)  # PH0 + one compare phase per bit
_T_INTRINSIC__ns = (0.5, 0.25, 0.75, 0.5)  # one latch delay per compare phase, deliberately unequal
_MIRROR_SCALE = 0.25
_E_FIXED__fJ = 100.0
_DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly so no compiled leaf is unrolled."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build_config(*, t_intrinsic__ns: tuple[float, ...] = _T_INTRINSIC__ns) -> RsCsaIadcConfig:
    return RsCsaIadcConfig(
        area_per_inst__um2=10.0,
        leakage_per_inst__uW=0.1,
        bits=_BITS,
        v_rail__V=_V_RAIL__V,
        t_phase__ns=_T_PHASE__ns,
        t_intrinsic__ns=t_intrinsic__ns,
        mirror_scale=_MIRROR_SCALE,
        e_fixed_per_op__fJ=_E_FIXED__fJ,
    )


def _build_adc(*, i_ph0_comp__uA: float = _I_PH0__uA) -> RsCsaIadc:
    """Owner-construct the readout: PH0 is injected, so there is no registry path."""
    adc = RsCsaIadc(
        config=_build_config(),
        policy=RsCsaIadcPolicy(),
        inst_shape=(1,),
        dtype=_DTYPE,
        T__K=300.0,
        i_ph0_comp__uA=i_ph0_comp__uA,
    )
    adc.eval()
    adc.fabricate()
    stamp_names(adc)  # the standalone converter is its own root, named ""
    return adc


def _ref() -> torch.Tensor:
    """The ONE reference current the compare phases scale, on its own tap axis."""
    return torch.tensor([_I_REF__uA], dtype=_DTYPE)


def _window_oracle__ns(bits: int) -> float:
    """Independent executed window: PH0 + the executed compare phases, cut at the latch."""
    return sum(_T_PHASE__ns[:bits]) + _T_INTRINSIC__ns[bits - 1]


def _convert_energy(adc: RsCsaIadc, i_in: torch.Tensor, *, bits: int = _BITS) -> float:
    """Total dynamic energy [fJ] of one convert under a fresh profiler."""
    with Profiler() as prof:
        adc.convert(i_in, _ref(), bits=bits)
    return Reporter(adc).total_dynamic_energy__fJ(prof)


def _energy_oracle(i_in__uA: float, *, bits: int = _BITS, i_ph0__uA: float = _I_PH0__uA) -> float:
    """Independent per-conversion energy: prorated E_fixed + the EXECUTED compare phases."""
    residue = max(i_in__uA - i_ph0__uA, 0.0)
    energy = _E_FIXED__fJ * _window_oracle__ns(bits) / _window_oracle__ns(_BITS)
    for phase in range(1, bits + 1):
        # Phase p resolves bit B - p, so it compares against that place value.
        i_phase_ref = (1 << (_BITS - phase)) * _I_REF__uA
        energy += _MIRROR_SCALE * _V_RAIL__V * _T_PHASE__ns[phase] * min(residue, i_phase_ref)
        if residue >= i_phase_ref:
            residue -= i_phase_ref
    return energy


# --- Surface + derived timing ---


def test_surface_and_injected_ph0() -> None:
    adc = _build_adc()
    assert adc.max_bits == _BITS
    assert adc.unsigned_range(_BITS) == (0, (1 << _BITS) - 1)
    assert adc.i_ph0_comp__uA == _I_PH0__uA


def test_conversion_window_is_derived_from_the_executed_phase_set() -> None:
    """`T_AC(b) = PH0 + the first b-1 compare phases + the last executed phase's OWN latch delay`."""
    adc = _build_adc()
    for bits in range(1, _BITS + 1):
        assert float(adc.t_conversion__ns(bits)) == pytest.approx(_window_oracle__ns(bits))
    # One bit runs PH0 and the FIRST compare phase's latch delay alone — the
    # phase that actually runs, not the one that would close a full conversion.
    assert float(adc.t_conversion__ns(1)) == pytest.approx(_T_PHASE__ns[0] + _T_INTRINSIC__ns[0])
    # At full resolution it is strictly shorter than the nominal phase sum: the
    # access closes at the latch, inside the last compare phase.
    assert float(adc.t_conversion__ns(_BITS)) < sum(_T_PHASE__ns)


def test_conversion_window_grows_strictly_with_bits() -> None:
    """Each extra bit adds one whole compare phase, and moves the latch to that phase's."""
    adc = _build_adc()
    windows = [float(adc.t_conversion__ns(bits)) for bits in range(1, _BITS + 1)]
    assert all(lo < hi for lo, hi in itertools.pairwise(windows)), windows
    for bits in range(2, _BITS + 1):
        grown = _T_PHASE__ns[bits - 1] + _T_INTRINSIC__ns[bits - 1] - _T_INTRINSIC__ns[bits - 2]
        assert windows[bits - 1] - windows[bits - 2] == pytest.approx(grown)


def test_conversion_window_rejects_unsupported_bits() -> None:
    """The window is defined only for a resolution the phase set can run."""
    adc = _build_adc()
    for bits in (0, _BITS + 1):
        with pytest.raises(ValueError, match=rf"require: bits \({bits}\) in \[1, max_bits \({_BITS}\)\]"):
            adc.t_conversion__ns(bits)


def test_latch_delay_is_one_per_compare_phase_and_fits_inside_it() -> None:
    """Each compare phase closes on its OWN latch, so the delay set is per phase."""
    with pytest.raises(ValueError, match="t_intrinsic__ns"):
        _build_config(t_intrinsic__ns=_T_INTRINSIC__ns[:-1])  # one short of the phase count
    # A delay longer than the phase it closes would run past the phase boundary.
    overrun = (*_T_INTRINSIC__ns[:-1], _T_PHASE__ns[-1] + 1.0)
    with pytest.raises(ValueError, match="t_intrinsic__ns"):
        _build_config(t_intrinsic__ns=overrun)
    # A delay that overruns a SHORTER earlier phase is caught at that phase.
    early_overrun = (_T_PHASE__ns[1] + 1.0, *_T_INTRINSIC__ns[1:])
    with pytest.raises(ValueError, match=r"t_intrinsic__ns\[0\]"):
        _build_config(t_intrinsic__ns=early_overrun)


# --- Quantizer laws ---


def test_uniform_quantize_matches_floor() -> None:
    adc = _build_adc()
    i_in = torch.tensor([1.6, 2.4, 3.0, 5.2, 7.0, 9.9], dtype=_DTYPE)
    code = adc.convert(i_in, _ref(), bits=_BITS)
    i_comp = (i_in - _I_PH0__uA).clamp(min=0.0)
    expected = torch.floor(i_comp / _I_REF__uA).clamp(0, (1 << _BITS) - 1).to(code.dtype)
    assert torch.equal(code, expected)


def test_monotone_non_decreasing() -> None:
    adc = _build_adc()
    i_in = torch.linspace(0.0, 10.0, 40, dtype=_DTYPE)
    code = adc.convert(i_in, _ref(), bits=_BITS)
    assert bool((code[1:] >= code[:-1]).all())


def test_at_or_below_ph0_is_zero() -> None:
    adc = _build_adc()
    i_in = torch.tensor([0.0, 0.3, _I_PH0__uA], dtype=_DTYPE)  # all <= i_ph0
    code = adc.convert(i_in, _ref(), bits=_BITS)
    assert torch.equal(code, torch.zeros_like(code))


def test_ph0_static_operand_independent() -> None:
    adc = _build_adc()
    # i_ph0 = 2 * i_ref -> convert(i + i_ph0) exceeds convert(i) by a fixed
    # 2 codes for every i whose compensated current stays in range.
    i_vals = torch.tensor([1.5, 2.0, 3.0, 4.0], dtype=_DTYPE)
    base = adc.convert(i_vals, _ref(), bits=_BITS)
    shifted = adc.convert(i_vals + _I_PH0__uA, _ref(), bits=_BITS)
    diff = (shifted - base).to(torch.long)
    assert torch.equal(diff, torch.full_like(diff, 2))


def test_bits_bounded_by_the_physical_resolution() -> None:
    """A resolution outside the physical one is rejected."""
    adc = _build_adc()
    i_in = torch.tensor([3.0], dtype=_DTYPE)
    with pytest.raises(ValueError, match=rf"require: bits \({_BITS + 1}\) in \[1, max_bits \({_BITS}\)\]"):
        adc.convert(i_in, _ref(), bits=_BITS + 1)
    with pytest.raises(ValueError, match=rf"require: bits \(0\) in \[1, max_bits \({_BITS}\)\]"):
        adc.convert(i_in, _ref(), bits=0)
    with pytest.raises(ValueError, match=rf"require: bits \({_BITS + 1}\) in \[1, max_bits \({_BITS}\)\]"):
        adc.unsigned_range(_BITS + 1)


def test_single_reference_input_at_every_bits() -> None:
    """The converter takes ONE reference current, whatever resolution is requested.

    The tap count is this circuit's own fact — it derives the whole ladder from
    that one current — so a multi-tap bank is rejected at every `bits`.
    """
    adc = _build_adc()
    i_in = torch.tensor([3.0], dtype=_DTYPE)
    ladder = _I_REF__uA * torch.arange(1, (1 << _BITS), dtype=_DTYPE)
    for bits in range(1, _BITS + 1):
        with pytest.raises(ValueError, match="n_taps"):
            adc.convert(i_in, ladder, bits=bits)
        adc.convert(i_in, _ref(), bits=bits)


def test_per_instance_reference_broadcasts_over_the_input() -> None:
    """The reference is per-instance: each position quantizes on its own step."""
    adc = _build_adc()
    i_in = torch.tensor([3.0, 3.0], dtype=_DTYPE)
    refs = torch.tensor([[_I_REF__uA], [2.0 * _I_REF__uA]], dtype=_DTYPE)
    code = adc.convert(i_in, refs, bits=_BITS)
    i_comp = (i_in - _I_PH0__uA).clamp(min=0.0)
    expected = torch.floor(i_comp / refs.squeeze(-1)).clamp(0, (1 << _BITS) - 1).to(code.dtype)
    assert torch.equal(code, expected)


def test_lowered_bits_drop_the_code_low_bits() -> None:
    """Equivalence law: `convert(bits=b) == convert(bits=B) >> (B - b)`.

    The whole ladder stays wired at every width — it is derived from the one
    reference, not from the requested resolution — so a lowered width widens the
    bin instead of moving the transfer. `b = 1` and `b = B` are both covered.
    """
    adc = _build_adc()
    i_in = torch.linspace(0.0, 10.0, 64, dtype=_DTYPE)
    full = adc.convert(i_in, _ref(), bits=_BITS)
    for bits in range(1, _BITS + 1):
        code = adc.convert(i_in, _ref(), bits=bits)
        assert code.dtype == full.dtype
        assert adc.unsigned_range(bits) == (0, (1 << bits) - 1)
        assert int(code.max()) <= (1 << bits) - 1
        assert torch.equal(code, full >> (_BITS - bits)), f"bits {bits}: {code.tolist()}"


def test_deterministic_in_training_mode() -> None:
    """No jitter is wired: `train()` converts exactly like `eval()`."""
    adc = _build_adc()
    i_in = torch.linspace(0.0, 9.0, 32, dtype=_DTYPE)
    eval_code = adc.convert(i_in, _ref(), bits=_BITS)
    adc.train()
    assert torch.equal(adc.convert(i_in, _ref(), bits=_BITS), eval_code)
    assert torch.equal(adc.convert(i_in, _ref(), bits=_BITS), eval_code)


# --- Energy laws ---


def test_energy_code_zero_is_exactly_e_fixed_at_full_resolution() -> None:
    """At full resolution the whole baseline is billed: a zero residue costs `E_fixed`."""
    adc = _build_adc()
    e = _convert_energy(adc, torch.tensor([_I_PH0__uA], dtype=_DTYPE))  # residue 0 -> code 0
    assert e == pytest.approx(_E_FIXED__fJ)


def test_energy_matches_per_phase_sar_oracle_at_every_bits() -> None:
    """Every conversion equals the prorated baseline plus the EXECUTED compare phases."""
    adc = _build_adc()
    for bits, i_in in itertools.product(range(1, _BITS + 1), (1.0, 1.6, 3.0, 5.2, 7.5, 9.9)):
        got = _convert_energy(adc, torch.tensor([i_in], dtype=_DTYPE), bits=bits)
        assert got == pytest.approx(_energy_oracle(i_in, bits=bits)), f"E mismatch at bits={bits}, i_in={i_in}"


def test_zero_residue_energy_ratio_is_the_window_ratio() -> None:
    """With no residue only the prorated baseline is left, so energy tracks the window."""
    adc = _build_adc()
    i_zero = torch.tensor([_I_PH0__uA], dtype=_DTYPE)  # residue 0 -> every compare term vanishes
    full__fJ = _convert_energy(adc, i_zero, bits=_BITS)
    for bits in range(1, _BITS + 1):
        ratio = _convert_energy(adc, i_zero, bits=bits) / full__fJ
        window_ratio = float(adc.t_conversion__ns(bits)) / float(adc.t_conversion__ns(_BITS))
        assert ratio == pytest.approx(window_ratio), f"baseline not prorated at bits={bits}"


def test_energy_grows_with_code() -> None:
    adc = _build_adc()
    e_zero = _convert_energy(adc, torch.tensor([_I_PH0__uA], dtype=_DTYPE))  # code 0
    e_low = _convert_energy(adc, torch.tensor([1.6], dtype=_DTYPE))  # code 1
    e_high = _convert_energy(adc, torch.tensor([9.9], dtype=_DTYPE))  # code 15
    assert e_zero < e_low < e_high


def test_energy_grows_strictly_with_bits() -> None:
    """Each extra bit adds one compare phase AND a longer prorated baseline."""
    adc = _build_adc()
    for i_in in (_I_PH0__uA, 9.9):  # no residue / residue in every phase
        energies = [_convert_energy(adc, torch.tensor([i_in], dtype=_DTYPE), bits=b) for b in range(1, _BITS + 1)]
        assert all(lo < hi for lo, hi in itertools.pairwise(energies)), f"i_in={i_in}: {energies}"


# --- Latency law ---


def test_reported_latency_is_the_executed_window() -> None:
    """`latency__ns` answers for the phase axis this converter owns — one conversion."""
    adc = _build_adc()
    for bits in range(1, _BITS + 1):
        assert adc.latency__ns(bits=bits) == pytest.approx(_window_oracle__ns(bits))


def test_reported_latency_rejects_a_resolution_the_phase_set_cannot_run() -> None:
    """No window outside the physical resolution."""
    adc = _build_adc()
    for bits in (0, _BITS + 1):
        with pytest.raises(ValueError, match=rf"require: bits \({bits}\) in \[1, max_bits \({_BITS}\)\]"):
            adc.latency__ns(bits=bits)
