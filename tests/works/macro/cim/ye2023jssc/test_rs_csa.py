"""CPU-only eager tests for the ye2023jssc RS-CSA readout.

The Reference-Subtracting CSA is a UNIFORM current quantizer with an
OWNER-INJECTED static PH0 compensation, a DERIVED per-resolution conversion
window, and a per-compare-phase SAR energy. Laws only (hand-written witness
config; tiny shapes; eager, dynamo disabled):

  * the conversion window is DERIVED, not configured, and follows the EXECUTED
    phases: ``t_conversion(b) == sum(t_phase[:b]) + t4_intrinsic`` — PH0, the
    first ``b - 1`` compare phases in full, and the last executed compare phase
    up to its comparator latch; it grows strictly with ``b`` and closes at the
    nominal ``sum(t_phase[:-1]) + t4_intrinsic`` at full resolution,
  * PH0 is a construction-time constant (no config field): it shifts every
    conversion by the same current, and an input at or below it reads code 0,
  * the compare phases are a uniform quantizer over the owner-supplied ladder:
    ``code == floor((i_in - i_ph0)+ / i_lsb)`` clamped to the 4-bit ceiling,
  * ``config.bits`` bounds the width a conversion may request and any ladder but
    the full ``2**bits - 1`` one is rejected; bit width is handled INSIDE the
    converter, which converts at full resolution and drops the code's low bits,
  * energy is ``E_fixed(b) + E_code(b)`` over the EXECUTED phases, with
    ``E_code = sum_{i<=b} mirror_scale * v_rail * min(residue_i, ref_radix[i]*i_lsb) * t_phase[i+1]``
    over the cumulative-subtraction residue (the latched REFS branches are
    rail-energy-neutral and are NOT billed) and the code-independent baseline
    prorated by the executed-window ratio, ``E_fixed(b) = E_fixed * T_AC(b) /
    T_AC(B)``: a zero-residue conversion costs the prorated baseline alone, so
    its energy ratio between two resolutions IS the window ratio,
  * latency is the executed window per serial round,
  * conversion is deterministic — no jitter is wired, so ``train()`` and
    ``eval()`` return the same codes.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.works.macro.cim.ye2023jssc.rscsa import (
    RsCsaIadc,
    RsCsaIadcConfig,
    RsCsaIadcPolicy,
)

_BITS = 4
_I_LSB__uA = 0.5
_REF_RADIX = (8, 4, 2, 1)
_I_PH0__uA = 1.0  # = 2 * i_lsb -> a clean 2-code static shift
_V_RAIL__V = 0.8
_T_PHASE__ns = (1.0, 2.0, 4.0, 8.0, 16.0)  # PH0 + one compare phase per bit
_T4_INTRINSIC__ns = 0.5
_MIRROR_SCALE = 0.25
_E_FIXED__fJ = 100.0
_DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly so no compiled leaf is unrolled."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build_config() -> RsCsaIadcConfig:
    return RsCsaIadcConfig(
        area_per_inst__um2=10.0,
        leakage_per_inst__uW=0.1,
        bits=_BITS,
        i_lsb__uA=_I_LSB__uA,
        ref_radix=_REF_RADIX,
        v_rail__V=_V_RAIL__V,
        t_phase__ns=_T_PHASE__ns,
        t4_intrinsic__ns=_T4_INTRINSIC__ns,
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
    return adc


def _taps() -> torch.Tensor:
    """The 15-tap ascending ladder ``c * i_lsb`` for ``c = 1 .. 15``."""
    return _I_LSB__uA * torch.arange(1, (1 << _BITS), dtype=_DTYPE)


def _window_oracle__ns(bits: int) -> float:
    """Independent executed window: PH0 + the executed compare phases, cut at the latch."""
    return sum(_T_PHASE__ns[:bits]) + _T4_INTRINSIC__ns


def _convert_energy(adc: RsCsaIadc, i_in: torch.Tensor, *, bits: int = _BITS) -> float:
    """Total dynamic energy [fJ] of one convert under a fresh profiler."""
    with NeuroxProfiler() as prof:
        adc.convert(i_in, _taps(), bits=bits)
    return prof.total_dynamic_energy__fJ


def _convert_latency(adc: RsCsaIadc, i_in: torch.Tensor, *, bits: int = _BITS) -> float:
    """Total latency [ns] of one convert under a fresh profiler."""
    with NeuroxProfiler() as prof:
        adc.convert(i_in, _taps(), bits=bits)
    return prof.total_latency__ns


def _energy_oracle(i_in__uA: float, *, bits: int = _BITS, i_ph0__uA: float = _I_PH0__uA) -> float:
    """Independent per-conversion energy: prorated E_fixed + the EXECUTED compare phases."""
    residue = max(i_in__uA - i_ph0__uA, 0.0)
    energy = _E_FIXED__fJ * _window_oracle__ns(bits) / _window_oracle__ns(_BITS)
    for phase, radix in enumerate(_REF_RADIX[:bits], start=1):
        i_ref = radix * _I_LSB__uA
        energy += _MIRROR_SCALE * _V_RAIL__V * _T_PHASE__ns[phase] * min(residue, i_ref)
        if residue >= i_ref:
            residue -= i_ref
    return energy


# --- Surface + derived timing ---


def test_surface_and_injected_ph0() -> None:
    adc = _build_adc()
    assert adc.max_bits == _BITS
    assert adc.unsigned_range(_BITS) == (0, (1 << _BITS) - 1)
    assert adc.i_ph0_comp__uA == _I_PH0__uA


def test_conversion_window_is_derived_from_the_executed_phase_set() -> None:
    """``T_AC(b) = PH0 + the first b-1 compare phases + the last executed phase's latch delay``."""
    adc = _build_adc()
    for bits in range(1, _BITS + 1):
        assert float(adc.t_conversion__ns(bits)) == pytest.approx(_window_oracle__ns(bits))
    # One bit runs PH0 and the latch delay alone.
    assert float(adc.t_conversion__ns(1)) == pytest.approx(_T_PHASE__ns[0] + _T4_INTRINSIC__ns)
    # At full resolution it is strictly shorter than the nominal phase sum: the
    # access closes at the latch, inside the last compare phase.
    assert float(adc.t_conversion__ns(_BITS)) < sum(_T_PHASE__ns)


def test_conversion_window_grows_strictly_with_bits() -> None:
    """Each extra bit adds one whole compare phase to the executed window."""
    adc = _build_adc()
    windows = [float(adc.t_conversion__ns(bits)) for bits in range(1, _BITS + 1)]
    assert all(lo < hi for lo, hi in itertools.pairwise(windows)), windows
    for bits in range(2, _BITS + 1):
        assert windows[bits - 1] - windows[bits - 2] == pytest.approx(_T_PHASE__ns[bits - 1])


def test_conversion_window_rejects_unsupported_bits() -> None:
    """The window is defined only for a resolution the phase set can run."""
    adc = _build_adc()
    for bits in (0, _BITS + 1):
        with pytest.raises(ValueError):
            adc.t_conversion__ns(bits)


# --- Quantizer laws ---


def test_uniform_quantize_matches_floor() -> None:
    adc = _build_adc()
    i_in = torch.tensor([1.6, 2.4, 3.0, 5.2, 7.0, 9.9], dtype=_DTYPE)
    code = adc.convert(i_in, _taps(), bits=_BITS)
    i_comp = (i_in - _I_PH0__uA).clamp(min=0.0)
    expected = torch.floor(i_comp / _I_LSB__uA).clamp(0, (1 << _BITS) - 1).to(code.dtype)
    assert torch.equal(code, expected)


def test_monotone_non_decreasing() -> None:
    adc = _build_adc()
    i_in = torch.linspace(0.0, 10.0, 40, dtype=_DTYPE)
    code = adc.convert(i_in, _taps(), bits=_BITS)
    assert bool((code[1:] >= code[:-1]).all())


def test_at_or_below_ph0_is_zero() -> None:
    adc = _build_adc()
    i_in = torch.tensor([0.0, 0.3, _I_PH0__uA], dtype=_DTYPE)  # all <= i_ph0
    code = adc.convert(i_in, _taps(), bits=_BITS)
    assert torch.equal(code, torch.zeros_like(code))


def test_ph0_static_operand_independent() -> None:
    adc = _build_adc()
    # i_ph0 = 2 * i_lsb -> convert(i + i_ph0) exceeds convert(i) by a fixed
    # 2 codes for every i whose compensated current stays in range.
    i_vals = torch.tensor([1.5, 2.0, 3.0, 4.0], dtype=_DTYPE)
    base = adc.convert(i_vals, _taps(), bits=_BITS)
    shifted = adc.convert(i_vals + _I_PH0__uA, _taps(), bits=_BITS)
    diff = (shifted - base).to(torch.long)
    assert torch.equal(diff, torch.full_like(diff, 2))


def test_bits_bounded_by_the_physical_resolution() -> None:
    """A resolution above the physical one, or a mis-sized ladder, is rejected."""
    adc = _build_adc()
    i_in = torch.tensor([3.0], dtype=_DTYPE)
    with pytest.raises(ValueError):
        adc.convert(i_in, _taps(), bits=_BITS + 1)
    with pytest.raises(ValueError):
        adc.convert(i_in, _taps(), bits=0)
    with pytest.raises(ValueError):
        adc.convert(i_in, _taps()[:-1], bits=_BITS)
    with pytest.raises(ValueError):
        adc.unsigned_range(_BITS + 1)


def test_full_ladder_required_at_every_bits() -> None:
    """The tap count states the READOUT's own width, never the requested one."""
    adc = _build_adc()
    i_in = torch.tensor([3.0], dtype=_DTYPE)
    for bits in range(1, _BITS):
        with pytest.raises(ValueError, match="n_taps"):
            adc.convert(i_in, _taps()[: (1 << bits) - 1], bits=bits)
        adc.convert(i_in, _taps(), bits=bits)


def test_lowered_bits_drop_the_code_low_bits() -> None:
    """Equivalence law: ``convert(bits=b) == convert(bits=B) >> (B - b)``.

    The full ladder stays wired at every width, so a lowered resolution widens
    the bin instead of moving the transfer. ``b = 1`` and ``b = B`` are both
    covered.
    """
    adc = _build_adc()
    i_in = torch.linspace(0.0, 10.0, 64, dtype=_DTYPE)
    full = adc.convert(i_in, _taps(), bits=_BITS)
    for bits in range(1, _BITS + 1):
        code = adc.convert(i_in, _taps(), bits=bits)
        assert code.dtype == full.dtype
        assert adc.unsigned_range(bits) == (0, (1 << bits) - 1)
        assert int(code.max()) <= (1 << bits) - 1
        assert torch.equal(code, full >> (_BITS - bits)), f"bits {bits}: {code.tolist()}"


def test_deterministic_in_training_mode() -> None:
    """No jitter is wired: ``train()`` converts exactly like ``eval()``."""
    adc = _build_adc()
    i_in = torch.linspace(0.0, 9.0, 32, dtype=_DTYPE)
    eval_code = adc.convert(i_in, _taps(), bits=_BITS)
    adc.train()
    assert torch.equal(adc.convert(i_in, _taps(), bits=_BITS), eval_code)
    assert torch.equal(adc.convert(i_in, _taps(), bits=_BITS), eval_code)


# --- Energy laws ---


def test_energy_code_zero_is_exactly_e_fixed_at_full_resolution() -> None:
    """At full resolution the whole baseline is billed: a zero residue costs ``E_fixed``."""
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


def test_latency_is_the_executed_window_per_serial_round() -> None:
    """One conversion bills its executed window; a batch serializes over the rounds."""
    adc = _build_adc()
    for bits in range(1, _BITS + 1):
        got = _convert_latency(adc, torch.tensor([3.0], dtype=_DTYPE), bits=bits)
        assert got == pytest.approx(_window_oracle__ns(bits))
    round_num = 5
    batched = _convert_latency(adc, torch.full((round_num,), 3.0, dtype=_DTYPE), bits=_BITS)
    assert batched == pytest.approx(round_num * _window_oracle__ns(_BITS))
