"""Laws for the scheme-local TMCSA phase-resolved conversion-billing module.

Hand-built tiny witness, eager, CPU. Five laws:

  * SHAPE LAW: ``inst_count`` derives from the ``(gn,)`` fabrication shape and
    the static PPA seats scale with it; ``max_bits`` derives from the
    phase-window list length; forward is billing-only (returns ``None``, value
    untouched).
  * LUT CONSISTENCY LAW (mandatory): the structural code -> reference-tap LUT
    equals the kernel SarIadc's ACTUAL ``_select_ref`` binary-search sequence,
    replayed step by step for EVERY final unsigned code under an all-off
    policy with clean margins; the replayed final code and a full ``convert``
    round-trip both reconcile.
  * PHASE-BILLING LAW (branch-tensor law): the recorded dynamic energy equals
    the hand-computed per-step formula on a tiny witness —
    ``sum_s v_dd * (3 * (i_sub + i_ref_path[s]) * t_ph2[s]
    + 2 * (i_sub + i_ref_path[s]) * t_ph3[s]) + e_fixed * bits`` per
    converted element, with ``i_ref_path[s]`` looked up from the final code.
  * LOWERED-BIT LAW: a ``b``-bit conversion truncates the max-bits search after
    ``b`` levels, so it bills the LEADING ``b`` phase windows at the up-shifted
    code — checked against the max-bits call with the trailing windows zeroed,
    not against a restated formula.
  * GUARDS: mismatched phase-window list lengths and negative entries are
    rejected at config time; a code/input shape mismatch, a wrong ladder tap
    count, and a ``bits`` outside ``[1, max_bits]`` are rejected at call time.
"""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog.current_adc import SarIadc, SarIadcConfig, SarIadcPolicy
from neurox.works.macro.cim.xue2020jssc.tmcsa import Tmcsa, TmcsaConfig, TmcsaPolicy

_DTYPE = torch.float64

# --- Tiny witness geometry ---
_GN = 2  # CIM-IO count
_SERIAL = 2  # column-MUX slots (ride the anonymous leading batch)
_BITS = 3

# --- Witness physics knobs (small explicit values, no code defaults) ---
_V_DD__V = 1.2  # non-unity so a dropped rail factor is caught
_T_PH2__NS = (0.5, 0.4, 0.3)
_T_PH3__NS = (0.9, 0.8, 0.7)
_E_FIXED__fJ = 1.25
_AREA_PER_INST__um2 = 2.0
_LEAKAGE_PER_INST__uW = 3.0

# Unit-step ascending ladder: tap index k carries value k + 1, so a selected
# reference value maps back to its tap index exactly (value - 1).
_LADDER = tuple(float(k + 1) for k in range((1 << _BITS) - 1))


def _config(
    *,
    t_ph2__ns: tuple[float, ...] = _T_PH2__NS,
    t_ph3__ns: tuple[float, ...] = _T_PH3__NS,
) -> TmcsaConfig:
    return TmcsaConfig(
        t_ph2_per_step__ns=t_ph2__ns,
        t_ph3_per_step__ns=t_ph3__ns,
        e_fixed_per_op__fJ=_E_FIXED__fJ,
        area_per_inst__um2=_AREA_PER_INST__um2,
        leakage_per_inst__uW=_LEAKAGE_PER_INST__uW,
    )


def _build(*, gn: int = _GN) -> Tmcsa:
    module = Tmcsa(
        config=_config(),
        policy=TmcsaPolicy(),
        inst_shape=(gn,),
        v_dd__V=_V_DD__V,
        dtype=_DTYPE,
    )
    module.eval()
    module.fabricate()
    stamp_names(module)  # the standalone module is its own root, named ""
    return module


def _build_kernel_adc() -> SarIadc:
    """All-off kernel SarIadc whose ``_select_ref`` sequence the LUT must replay."""
    adc = SarIadc(
        config=SarIadcConfig(
            bits=_BITS,
            margin_gain=3.0,
            e_fixed_per_op__fJ=0.0,
            v_rail__V=0.0,
            t_conduct_per_step__ns=(0.0,) * _BITS,
            step_latency__ns=(1.0,) * _BITS,
            comparator_offset_sigma__uA=0.0,
            coupling_mismatch_sigma__uA=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=SarIadcPolicy(comparator_offset=False, coupling_mismatch=False),
        inst_shape=(1,),
        dtype=_DTYPE,
        T__K=300.0,
    )
    adc.eval()
    adc.fabricate()
    return adc


# ---------------------------------------------------------------------------
# Shape law
# ---------------------------------------------------------------------------


def test_shape_law() -> None:
    """inst_count = gn; bits = phase-window length; static PPA scales with inst_count."""
    module = _build()
    assert module.inst_shape == (_GN,)
    assert module.inst_count == _GN
    assert module.max_bits == _BITS
    assert module.area__um2 == pytest.approx(_AREA_PER_INST__um2 * _GN)
    assert module.leakage__uW == pytest.approx(_LEAKAGE_PER_INST__uW * _GN)


def test_forward_is_billing_only() -> None:
    """forward returns None (the value conversion lives in the kernel ADC)."""
    module = _build()
    # Shape: [1, serial, gn]
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[1, 2], [0, 6]]], dtype=torch.long)
    refs = torch.tensor(_LADDER, dtype=_DTYPE)
    assert module(i_sub, code, refs, bits=_BITS) is None


# ---------------------------------------------------------------------------
# LUT consistency law (F2c — mandatory)
# ---------------------------------------------------------------------------


def _replay_select_ref_sequence(adc: SarIadc, i_in: torch.Tensor, refs: torch.Tensor) -> tuple[list[int], int]:
    """Replay the kernel binary search step by step; return (tap sequence, final code).

    Mirrors ``SarIadc._convert_impl`` exactly under all-off (clean margins,
    zero fabricated offset): per step, the selected reference comes from the
    kernel's own ``_select_ref`` on the PARTIAL code, and the decision bit is
    the clean threshold ``i_in > i_ref``. The unit-step ladder makes the
    selected tap index recoverable from the reference value (value - 1).
    """
    max_bits = adc.max_bits
    ref_b = torch.broadcast_to(refs, (*i_in.shape, refs.shape[-1]))
    code = torch.zeros_like(i_in, dtype=torch.long)
    taps: list[int] = []
    for step in range(max_bits):
        i_ref = adc._select_ref(ref_b, code, step, max_bits)
        taps.append(round(float(i_ref)) - 1)
        bit = (i_in - i_ref) > 0.0
        code = adc._set_bit(code, step, bit, max_bits)
    return taps, int(code)


def test_lut_matches_kernel_select_ref_sequence_for_every_code() -> None:
    """For EVERY final code the structural LUT equals the replayed kernel tap sequence."""
    module = _build()
    adc = _build_kernel_adc()
    refs = torch.tensor(_LADDER, dtype=_DTYPE)
    for c in range(1 << _BITS):
        # i_in strictly between ladder values c and c+1 lands on final code c.
        i_in = torch.tensor(c + 0.5, dtype=_DTYPE)
        taps, final_code = _replay_select_ref_sequence(adc, i_in, refs)
        assert final_code == c, f"replay landed on code {final_code}, expected {c}"
        # The full kernel convert reconciles with the replay (all-off, clean margins).
        assert int(adc.convert(i_in.reshape(1), refs, bits=_BITS)) == c
        assert module._ref_tap_lut[c].tolist() == taps, (
            f"LUT row for code {c}: {module._ref_tap_lut[c].tolist()} != replayed kernel sequence {taps}"
        )


def test_lut_closed_form_anchor() -> None:
    """The documented closed form pins its worked example: B=3, c=5 -> taps (3, 5, 4)."""
    module = _build()
    assert module._ref_tap_lut.shape == (1 << _BITS, _BITS)
    assert module._ref_tap_lut[5].tolist() == [3, 5, 4]


# ---------------------------------------------------------------------------
# Phase-billing law
# ---------------------------------------------------------------------------


def test_phase_billing_law_hand_computed() -> None:
    """Recorded energy == the hand-computed PH2/PH3 per-step formula."""
    module = _build()
    # Shape: [1, serial, gn]
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[1, 2], [0, 6]]], dtype=torch.long)
    refs = torch.tensor(_LADDER, dtype=_DTYPE)

    with Profiler() as prof, torch.no_grad():
        module(i_sub, code, refs, bits=_BITS)

    expected = 0.0
    for i_val, c_val in zip(i_sub.flatten().tolist(), code.flatten().tolist(), strict=True):
        for s in range(_BITS):
            i_ref = _LADDER[int(module._ref_tap_lut[c_val, s])]
            i_ph2 = 3.0 * (i_val + i_ref)  # PH2: inputs (1x each) + internal P3/P4 (2x each)
            i_ph3 = 2.0 * (i_val + i_ref)  # PH3: internal only; 2x splits into two 1x sinks
            expected += _V_DD__V * (i_ph2 * _T_PH2__NS[s] + i_ph3 * _T_PH3__NS[s]) + _E_FIXED__fJ

    reporter = Reporter(module)
    assert reporter.total_dynamic_energy__fJ(prof) == pytest.approx(expected, rel=1e-12)
    # One un-channelled record per forward — the module's own report row.
    assert len(prof.records) == 1
    assert prof.records[0].channel is None
    assert reporter.by_name(prof) == {"": pytest.approx(expected, rel=1e-12)}


def _bill(module: Tmcsa, i_sub: torch.Tensor, code: torch.Tensor, refs: torch.Tensor, *, bits: int) -> float:
    """Total dynamic energy [fJ] one billing call records."""
    with Profiler() as prof, torch.no_grad():
        module(i_sub, code, refs, bits=bits)
    return Reporter(module).total_dynamic_energy__fJ(prof)


def test_lowered_bits_bills_the_leading_steps_at_the_up_shifted_code() -> None:
    """A ``b``-bit conversion bills the FIRST ``b`` steps at the up-shifted code.

    Bits ``b`` truncates the max-bits search after ``b`` levels over the same
    full ladder, so it runs the leading ``b`` steps of the max-bits search and
    lands on the max-bits code shifted down by ``B - b``. Cross-checked against the
    max-bits call itself rather than a restated formula: run the up-shifted code
    at max bits with the trailing phase windows zeroed — that isolates the same
    leading steps — and discount the ``e_fixed`` of the steps a ``b``-bit
    conversion never runs.
    """
    bits = _BITS - 1
    shift = _BITS - bits
    # Shape: [1, serial, gn]
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)  # b-bit codes
    refs = torch.tensor(_LADDER, dtype=_DTYPE)  # always the FULL max-bits ladder

    lowered = _bill(_build(), i_sub, code, refs, bits=bits)

    leading_only = Tmcsa(
        config=_config(
            t_ph2__ns=_T_PH2__NS[:bits] + (0.0,) * shift,
            t_ph3__ns=_T_PH3__NS[:bits] + (0.0,) * shift,
        ),
        policy=TmcsaPolicy(),
        inst_shape=(_GN,),
        v_dd__V=_V_DD__V,
        dtype=_DTYPE,
    )
    leading_only.eval()
    leading_only.fabricate()
    stamp_names(leading_only)
    full_width = _bill(leading_only, i_sub, code << shift, refs, bits=_BITS)

    assert lowered == pytest.approx(full_width - _E_FIXED__fJ * shift * i_sub.numel(), rel=1e-12)


def test_forward_rejects_bits_outside_the_phase_windows() -> None:
    """``bits`` is the width actually converted: it must lie in ``[1, max_bits]``."""
    module = _build()
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[1, 2], [0, 6]]], dtype=torch.long)
    refs = torch.tensor(_LADDER, dtype=_DTYPE)
    for bad in (0, _BITS + 1):
        with pytest.raises(ValueError, match="bits"):
            module(i_sub, code, refs, bits=bad)


def test_billing_outside_profiler_is_silent() -> None:
    """forward outside a profiler records nothing and raises nothing."""
    module = _build()
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[1, 2], [0, 6]]], dtype=torch.long)
    module(i_sub, code, torch.tensor(_LADDER, dtype=_DTYPE), bits=_BITS)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_config_rejects_mismatched_or_negative_phase_windows() -> None:
    """Phase-window lists must be same-length, non-empty, and non-negative."""
    with pytest.raises(ValueError, match="t_ph3_per_step__ns"):
        _config(t_ph2__ns=(0.5, 0.4, 0.3), t_ph3__ns=(0.9, 0.8))
    with pytest.raises(ValueError, match="t_ph2_per_step__ns"):
        _config(t_ph2__ns=(), t_ph3__ns=())
    with pytest.raises(ValueError, match="t_ph2_per_step__ns"):
        _config(t_ph2__ns=(0.5, -0.4, 0.3), t_ph3__ns=(0.9, 0.8, 0.7))


def test_forward_rejects_shape_and_tap_mismatches() -> None:
    """A code/input shape mismatch and a wrong ladder tap count are rejected."""
    module = _build()
    i_sub = torch.tensor([[[1.5, 2.5], [0.5, 6.5]]], dtype=_DTYPE)
    code = torch.tensor([[[1, 2], [0, 6]]], dtype=torch.long)
    refs = torch.tensor(_LADDER, dtype=_DTYPE)
    with pytest.raises(ValueError, match=r"code\.shape"):
        module(i_sub, code[..., :1], refs, bits=_BITS)
    with pytest.raises(ValueError, match="n_taps"):
        module(i_sub, code, refs[:-1], bits=_BITS)
