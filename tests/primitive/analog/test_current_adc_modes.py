"""Multi-mode 2-D references on the current-ADC side + family template method.

``SarCurrentAdcConfig.ref_levels__uA`` and ``CurrentReferenceConfig.i_refs__uA``
are 2-D ``[mode][tap]`` banks; ``SarCurrentAdc._convert_impl`` consumes
``adc_operation_point.adc_mode`` to select the ladder row. These tests pin:

- 2-D config validation: per-row exact ``2 ** n_bits - 1`` length, strictly
  increasing rows, and the flat-tuple single-mode canonicalization;
- mode-selection correctness: two artificial ladders yield different codes
  for the same input, each matching the single-mode reference conversion;
- the ``adc_mode`` bounds error and the ``adc_bits == n_bits`` validation;
- quasi-static mode semantics: selecting a mode row changes no energy or
  latency accounting versus a single-mode ADC holding the same ladder;
- ``CurrentAdc.convert`` template method: probe-off equivalence with
  ``_convert_impl`` and probe capture of input, code, and operating point.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.prober import AdcProber, Prober
from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.analog.current_adc import (
    SarCurrentAdc,
    SarCurrentAdcConfig,
    SarCurrentAdcPolicy,
)

# Two artificial 3-bit ladders (7 thresholds each): a unit-step ladder and a
# 10x-spread ladder, so the same input lands on different codes per mode.
_LADDER_A = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
_LADDER_B = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)


def _config(ref_levels__uA: tuple[tuple[float, ...], ...] | tuple[float, ...]) -> SarCurrentAdcConfig:
    return SarCurrentAdcConfig(
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        n_bits=3,
        margin_gain=3.0,
        ref_levels__uA=ref_levels__uA,
        e_fixed_per_op__fJ=7.0,
        step_latency__ns=(3.0, 3.0, 3.0),
        comparator_offset_sigma__uA=0.0,
        coupling_mismatch_sigma__uA=0.0,
        mirror_mismatch_sigma_relative=0.0,
    )


def _build(ref_levels__uA: tuple[tuple[float, ...], ...] | tuple[float, ...], device: torch.device) -> SarCurrentAdc:
    adc = SarCurrentAdc(
        config=_config(ref_levels__uA),
        policy=SarCurrentAdcPolicy(
            comparator_offset=False,
            replica_threshold_variation=False,
            mirror_mismatch=False,
            coupling_mismatch=False,
        ),
        inst_shape=(1,),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.to(device)
    adc.eval()
    return adc


def _op(adc_mode: int, adc_bits: int = 3) -> AdcOperationPoint:
    return AdcOperationPoint(adc_mode=adc_mode, adc_bits=adc_bits)


# ---------------------------------------------------------------------------
# 2-D config validation
# ---------------------------------------------------------------------------


def test_config_rejects_bad_2d_banks() -> None:
    """Wrong row length, non-increasing rows, and empty banks are rejected."""
    for bad in (
        (),  # no modes
        (_LADDER_A[:6],),  # row too short (6 != 7)
        ((*_LADDER_A, 8.0),),  # row too long (8 != 7)
        ((7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0),),  # decreasing row
        ((1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),),  # not strictly increasing
        (_LADDER_A, (*_LADDER_B[:6], 60.0)),  # second row not strictly increasing
    ):
        with pytest.raises(ValueError):
            _config(bad)
    # Both rows well-formed: accepted, mode_num == 2.
    assert _config((_LADDER_A, _LADDER_B)).mode_num == 2


def test_flat_tuple_canonicalizes_to_single_mode() -> None:
    """A flat in-code ladder canonicalizes to one mode row (single-mode shorthand)."""
    config = _config(_LADDER_A)
    assert config.ref_levels__uA == (_LADDER_A,)
    assert config.mode_num == 1


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_mode_selection_changes_codes(device: torch.device) -> None:
    """The same input quantizes differently under the two artificial ladders."""
    adc = _build((_LADDER_A, _LADDER_B), device)
    i_in = torch.tensor([0.5, 4.5, 35.0, 100.0], dtype=torch.float64, device=device)

    code_a = adc.convert(i_in, adc_operation_point=_op(0))
    code_b = adc.convert(i_in, adc_operation_point=_op(1))

    assert torch.equal(code_a, torch.tensor([0, 4, 7, 7], device=device))
    assert torch.equal(code_b, torch.tensor([0, 0, 3, 7], device=device))
    assert not torch.equal(code_a, code_b)


def test_mode_row_matches_single_mode_reference(device: torch.device) -> None:
    """Each mode row reproduces the codes of a single-mode ADC holding that ladder."""
    multi = _build((_LADDER_A, _LADDER_B), device)
    i_in = torch.linspace(0.0, 80.0, 41, dtype=torch.float64, device=device)
    for mode, ladder in enumerate((_LADDER_A, _LADDER_B)):
        single = _build((ladder,), device)
        assert torch.equal(
            multi.convert(i_in, adc_operation_point=_op(mode)),
            single.convert(i_in, adc_operation_point=_op(0)),
        )


def test_mode_out_of_bounds_raises(device: torch.device) -> None:
    """``adc_mode`` outside ``[0, mode_num)`` is rejected."""
    adc = _build((_LADDER_A, _LADDER_B), device)
    assert adc.mode_num == 2
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for mode in (-1, 2):
        with pytest.raises(ValueError, match="adc_mode"):
            adc.convert(i_in, adc_operation_point=_op(mode))


def test_adc_bits_must_equal_n_bits(device: torch.device) -> None:
    """``adc_bits`` differing from the physical ``n_bits`` is rejected."""
    adc = _build((_LADDER_A,), device)
    i_in = torch.tensor([1.5], dtype=torch.float64, device=device)
    for bits in (1, 2, 4):
        with pytest.raises(ValueError, match="adc_bits"):
            adc.convert(i_in, adc_operation_point=_op(0, adc_bits=bits))


# ---------------------------------------------------------------------------
# Quasi-static energy semantics
# ---------------------------------------------------------------------------


def test_mode_selection_is_energy_free(device: torch.device) -> None:
    """Selecting a mode row emits the same energy/latency as a single-mode ADC.

    Mode selection is quasi-static: no per-conversion switching energy, so
    converting on row 1 of a 2-mode bank must be event-identical (count,
    energy, latency) to a single-mode ADC holding the same ladder.
    """
    i_in = torch.tensor([0.5, 4.5, 35.0, 100.0], dtype=torch.float64, device=device)

    multi = _build((_LADDER_A, _LADDER_B), device)
    with NeuroxProfiler() as p_multi:
        multi.convert(i_in, adc_operation_point=_op(1))

    single = _build((_LADDER_B,), device)
    with NeuroxProfiler() as p_single:
        single.convert(i_in, adc_operation_point=_op(0))

    assert len(p_multi.energy_events) == len(p_single.energy_events) == 1
    assert len(p_multi.latency_events) == len(p_single.latency_events) == 1
    assert p_multi.total_dynamic_energy__fJ == pytest.approx(p_single.total_dynamic_energy__fJ)
    assert p_multi.total_latency__ns == pytest.approx(p_single.total_latency__ns)


# ---------------------------------------------------------------------------
# Family template method (probe side channel)
# ---------------------------------------------------------------------------


def test_probe_off_convert_matches_convert_impl(device: torch.device) -> None:
    """Without an active prober the template equals the leaf conversion body."""
    adc = _build((_LADDER_A, _LADDER_B), device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)
    op = _op(1)

    assert not Prober._active_stack
    via_template = adc.convert(i_in, adc_operation_point=op)
    direct = adc._convert_impl(i_in, adc_operation_point=op)
    assert torch.equal(via_template, direct)


def test_probe_capture_carries_input_code_and_op_point(device: torch.device) -> None:
    """An active prober records the call's input, code, and operating point."""
    adc = _build((_LADDER_A, _LADDER_B), device)
    i_in = torch.tensor([0.5, 4.5, 35.0], dtype=torch.float64, device=device)
    op = _op(1)

    with AdcProber() as prober:
        code = adc.convert(i_in, adc_operation_point=op)

    records = prober.convert_records()
    assert len(records) == 1
    module, tensors = records[0]
    assert module is adc
    assert torch.equal(tensors["i_in__uA"], i_in)
    assert torch.equal(tensors["code"], code)
    assert int(tensors["adc_mode"]) == 1
    assert int(tensors["adc_bits"]) == 3
