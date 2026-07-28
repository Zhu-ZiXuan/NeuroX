"""CPU-only eager tests for the ye2023jssc RS-CSA readout.

The Reference-Subtracting CSA is a UNIFORM current quantizer with an
OWNER-INJECTED static PH0 compensation, a DERIVED conversion window, and a
per-compare-phase SAR energy. Laws only (hand-written witness config; tiny
shapes; eager, dynamo disabled):

  * the conversion window is DERIVED, not configured:
    ``t_conversion == sum(t_phase[:-1]) + t4_intrinsic`` — the access closes at
    the last comparator latch, inside the last compare phase,
  * PH0 is a construction-time constant (no config field): it shifts every
    conversion by the same current, and an input at or below it reads code 0,
  * the compare phases are a uniform quantizer over the owner-supplied ladder:
    ``code == floor((i_in - i_ph0)+ / i_lsb)`` clamped to the 4-bit ceiling,
  * the readout runs ONE fixed phase set, so ``config.bits`` bounds the width a
    conversion may request and a ladder of the wrong length is rejected; below
    that width a DECIMATED ladder drops the code's low bits,
  * energy is ``E_fixed + E_code`` with
    ``E_code = sum_i mirror_scale * v_rail * min(residue_i, ref_radix[i]*i_lsb) * t_phase[i+1]``
    over the cumulative-subtraction residue (the latched REFS branches are
    rail-energy-neutral and are NOT billed): code 0 costs exactly ``E_fixed``,
  * conversion is deterministic — no jitter is wired, so ``train()`` and
    ``eval()`` return the same codes.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.macro.cim import decimate_references
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


def _convert_energy(adc: RsCsaIadc, i_in: torch.Tensor) -> float:
    """Total dynamic energy [fJ] of one convert under a fresh profiler."""
    with NeuroxProfiler() as prof:
        adc.convert(i_in, _taps(), bits=_BITS)
    return prof.total_dynamic_energy__fJ


def _energy_oracle(i_in__uA: float, *, i_ph0__uA: float = _I_PH0__uA) -> float:
    """Independent per-conversion energy: E_fixed + the per-compare-phase SAR term."""
    residue = max(i_in__uA - i_ph0__uA, 0.0)
    energy = _E_FIXED__fJ
    for phase, radix in enumerate(_REF_RADIX, start=1):
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


def test_conversion_window_is_derived_from_the_phase_set() -> None:
    """``T_AC = PH0 + every compare phase but the last + the last phase's intrinsic latch delay``."""
    adc = _build_adc()
    assert float(adc.t_conversion__ns) == pytest.approx(sum(_T_PHASE__ns[:-1]) + _T4_INTRINSIC__ns)
    # It is strictly shorter than the nominal phase sum: the access closes at the latch.
    assert float(adc.t_conversion__ns) < sum(_T_PHASE__ns)


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


def test_decimated_ladder_drops_the_code_low_bits() -> None:
    """At ``bits < config.bits`` a decimated ladder yields the max-bits code shifted right.

    The analog machine is one operating point: the owner decimates the shared
    ladder, so the conversion widens its bin instead of moving its transfer.
    """
    adc = _build_adc()
    i_in = torch.linspace(0.0, 10.0, 64, dtype=_DTYPE)
    full = adc.convert(i_in, _taps(), bits=_BITS).to(torch.long)
    for bits in range(1, _BITS + 1):
        refs = decimate_references(_taps(), adc_max_bits=_BITS, adc_bits=bits)
        assert refs.shape[-1] == (1 << bits) - 1
        code = adc.convert(i_in, refs, bits=bits).to(torch.long)
        assert adc.unsigned_range(bits) == (0, (1 << bits) - 1)
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


def test_energy_code_zero_is_exactly_e_fixed() -> None:
    adc = _build_adc()
    e = _convert_energy(adc, torch.tensor([_I_PH0__uA], dtype=_DTYPE))  # residue 0 -> code 0
    assert e == pytest.approx(_E_FIXED__fJ)


def test_energy_matches_per_phase_sar_oracle() -> None:
    """Every conversion equals ``E_fixed + sum_i k*v_rail*min(residue_i, I_REF_i)*t_phase_i``."""
    adc = _build_adc()
    for i_in in (1.0, 1.6, 3.0, 5.2, 7.5, 9.9):
        got = _convert_energy(adc, torch.tensor([i_in], dtype=_DTYPE))
        assert got == pytest.approx(_energy_oracle(i_in)), f"E mismatch at i_in={i_in}"


def test_energy_grows_with_code() -> None:
    adc = _build_adc()
    e_zero = _convert_energy(adc, torch.tensor([_I_PH0__uA], dtype=_DTYPE))  # code 0
    e_low = _convert_energy(adc, torch.tensor([1.6], dtype=_DTYPE))  # code 1
    e_high = _convert_energy(adc, torch.tensor([9.9], dtype=_DTYPE))  # code 15
    assert e_zero < e_low < e_high
