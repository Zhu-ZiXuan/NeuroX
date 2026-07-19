"""Reference sources: multi-tap state, tolerance/noise, and the PPA-only contract.

``VoltageReference`` / ``CurrentReference`` are behavioural reference
sources: they carry static PPA (area + leakage) and hand out the actual
tap values through a snap, but perform no computation and emit no dynamic
energy or latency. ``VoltageReference`` holds a flat unordered tap tuple;
``CurrentReference`` holds a 2-D ``[mode][tap]`` bank of strictly
increasing equal-length rows whose mode selection is quasi-static. These
tests pin:

- multi-tap config loads a TOML array straight into the tuple field
  (flat for voltage, nested for current);
- 2-D validation on the current bank: strictly increasing rows, equal
  row lengths, non-negative taps, and the flat-tuple single-mode
  canonicalization for in-code construction;
- ``snapshot`` is deterministic with both nonidealities off and matches
  the nominal taps, broadcast to ``(*inst_shape, *tap_shape)``;
- the ``v_ref__V`` / ``i_ref__uA`` accessor returns every tap from a
  snap (the encapsulated read path consumers use instead of the buffer);
- ``tolerance`` is a per-die static spread fixed at ``fabricate`` time;
  ``noise`` is resampled every ``snapshot``;
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` + ``snapshot`` emit zero
  energy / latency events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.current_reference import (
    CurrentReference,
    CurrentReferenceConfig,
    CurrentReferencePolicy,
)
from neurox.primitive.analog.voltage_reference import (
    VoltageReference,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
)


@dataclass(frozen=True)
class _Case:
    """One reference-source family under test."""

    ref_cls: Any
    config_cls: Any
    policy_cls: Any
    field: str
    accessor: str
    taps: Any  # config value (flat for voltage, [mode][tap] for current)
    tap_shape: tuple[int, ...]  # trailing tap-tensor shape
    toml_array: str  # TOML literal for the tap field


_VOLTAGE = _Case(
    ref_cls=VoltageReference,
    config_cls=VoltageReferenceConfig,
    policy_cls=VoltageReferencePolicy,
    field="v_refs__V",
    accessor="v_ref__V",
    taps=(0.6, 1.2, 0.3),
    tap_shape=(3,),
    toml_array="[0.6, 1.2, 0.3]",
)
_CURRENT = _Case(
    ref_cls=CurrentReference,
    config_cls=CurrentReferenceConfig,
    policy_cls=CurrentReferencePolicy,
    field="i_refs__uA",
    accessor="i_ref__uA",
    taps=((1.0, 5.0, 20.0), (2.0, 6.0, 25.0)),
    tap_shape=(2, 3),
    toml_array="[[1.0, 5.0, 20.0], [2.0, 6.0, 25.0]]",
)
_CASES = [_VOLTAGE, _CURRENT]
_IDS = ["voltage", "current"]


def _config(case: _Case, **overrides: Any) -> Any:
    base = {
        case.field: case.taps,
        "tolerance_sigma_relative": 0.0,
        "noise_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return case.config_cls(**{**base, **overrides})


def _make(
    case: _Case,
    *,
    inst_shape: tuple[int, ...] = (),
    tolerance: bool = False,
    noise: bool = False,
    tol_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    area: float = 0.0,
    leakage: float = 0.0,
) -> Any:
    config = _config(
        case,
        tolerance_sigma_relative=tol_sigma,
        noise_sigma_relative=noise_sigma,
        area_per_inst__um2=area,
        leakage_per_inst__uW=leakage,
    )
    return case.ref_cls(
        config=config,
        policy=case.policy_cls(tolerance=tolerance, noise=noise),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_validation_rejects_bad_config(case: _Case) -> None:
    """Empty taps, negative taps, and negative sigmas/PPA are rejected."""
    for override in (
        {case.field: ()},  # empty tap list
        {"tolerance_sigma_relative": -1e-3},
        {"noise_sigma_relative": -1e-3},
        {"area_per_inst__um2": -1.0},
        {"leakage_per_inst__uW": -1.0},
    ):
        with pytest.raises(ValueError):
            _config(case, **override)


def test_voltage_taps_are_unordered_and_zero_ok() -> None:
    """Voltage taps stay a flat unordered tuple; a 0 tap (ground rail) is accepted."""
    with pytest.raises(ValueError):
        _config(_VOLTAGE, v_refs__V=(0.6, -1.0))  # negative tap
    _config(_VOLTAGE, v_refs__V=(0.6, 0.0))


def test_current_bank_2d_validation() -> None:
    """The current bank enforces strictly increasing, equal-length, non-negative rows."""
    for bad in (
        ((0.6, -1.0),),  # negative tap (also non-increasing)
        ((2.0, 1.0, 3.0),),  # non-increasing row
        ((1.0, 1.0),),  # not strictly increasing
        ((1.0, 2.0), (1.0, 2.0, 3.0)),  # unequal row lengths
        ((),),  # empty row
    ):
        with pytest.raises(ValueError):
            _config(_CURRENT, i_refs__uA=bad)
    # A 0 first tap denotes a ground/rail reference and is accepted.
    _config(_CURRENT, i_refs__uA=((0.0, 0.6),))


def test_current_flat_tuple_canonicalizes_to_single_mode() -> None:
    """A flat in-code tap tuple canonicalizes to one mode row (single-mode shorthand)."""
    config = _config(_CURRENT, i_refs__uA=(1.0, 5.0, 20.0))
    assert config.i_refs__uA == ((1.0, 5.0, 20.0),)
    assert config.mode_num == 1
    assert config.tap_num == 3


def test_current_mode_tap_counts() -> None:
    """``mode_num`` / ``tap_num`` report the bank geometry on config and module."""
    ref = _make(_CURRENT)
    assert ref.config.mode_num == 2
    assert ref.config.tap_num == 3
    assert ref.mode_num == 2
    assert ref.tap_num == 3


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_all_off_snapshot_matches_nominal(case: _Case) -> None:
    """With both nonidealities off, ``snapshot`` is the deterministic nominal taps."""
    ref = _make(case)
    snap_a = ref.snapshot()
    snap_b = ref.snapshot()
    out_a = getattr(snap_a, case.field)
    nominal = torch.tensor(case.taps, dtype=torch.float64)

    assert out_a.shape == case.tap_shape
    torch.testing.assert_close(out_a, nominal)
    torch.testing.assert_close(getattr(snap_b, case.field), nominal)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_inst_shape_broadcasts_taps(case: _Case) -> None:
    """A non-scalar ``inst_shape`` yields ``(*inst_shape, *tap_shape)`` taps."""
    ref = _make(case, inst_shape=(2, 3))
    out = getattr(ref.snapshot(), case.field)
    nominal = torch.tensor(case.taps, dtype=torch.float64)
    assert out.shape == (2, 3, *case.tap_shape)
    torch.testing.assert_close(out, nominal.expand(2, 3, *case.tap_shape))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_accessor_reads_taps_from_snap(case: _Case) -> None:
    """The ``v_ref__V`` / ``i_ref__uA`` accessor returns every tap from a snap."""
    ref = _make(case, inst_shape=(2, 3), noise=True, noise_sigma=0.05)
    ref.fabricate()
    accessor = getattr(ref, case.accessor)

    torch.manual_seed(0)
    snap = ref.snapshot()
    out = accessor(snap)

    # Encapsulated read: returns the snap's own (noised) tensor, full tap set.
    assert out.shape == (2, 3, *case.tap_shape)
    assert torch.equal(out, getattr(snap, case.field))
    # Pure: re-reading the same snap never resamples.
    assert torch.equal(accessor(snap), out)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_tolerance_is_static_per_die(case: _Case) -> None:
    """``tolerance`` is fixed at fabricate time and resampled only by ``fabricate``."""
    nominal = torch.tensor(case.taps, dtype=torch.float64)
    ref = _make(case, inst_shape=(8,), tolerance=True, tol_sigma=0.1)

    # Before fabricate(): the actual buffer is the untouched nominal.
    torch.testing.assert_close(getattr(ref, case.field), nominal.expand(8, *case.tap_shape))

    torch.manual_seed(0)
    ref.fabricate()
    fab1 = getattr(ref, case.field).clone()
    assert fab1.shape == (8, *case.tap_shape)
    assert not torch.allclose(fab1, nominal.expand(8, *case.tap_shape))

    # noise is off, so repeated snapshots return the same per-die taps.
    torch.testing.assert_close(getattr(ref.snapshot(), case.field), fab1)
    torch.testing.assert_close(getattr(ref.snapshot(), case.field), fab1)

    # A second fabricate() resamples the static spread.
    torch.manual_seed(1)
    ref.fabricate()
    assert not torch.allclose(getattr(ref, case.field), fab1)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_noise_resamples_every_snapshot(case: _Case) -> None:
    """``noise`` makes consecutive snapshots differ."""
    ref = _make(case, inst_shape=(4,), noise=True, noise_sigma=0.05)
    torch.manual_seed(0)
    first = getattr(ref.snapshot(), case.field)
    second = getattr(ref.snapshot(), case.field)
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_static_ppa_and_no_dynamic_events(case: _Case) -> None:
    """Static PPA scales by ``inst_count``; fabricate/snapshot emit no events."""
    ref = _make(
        case, inst_shape=(3,), tolerance=True, noise=True, tol_sigma=0.1, noise_sigma=0.05, area=2.0, leakage=0.5
    )

    assert ref.area__um2 == pytest.approx(2.0 * 3)
    assert ref.leakage__uW == pytest.approx(0.5 * 3)

    records = NeuroxProfiler.collect_static(ref)
    assert len(records) == 1
    assert records[0].area__um2 == pytest.approx(2.0 * 3)
    assert records[0].leakage_power__uW == pytest.approx(0.5 * 3)

    with NeuroxProfiler() as p:
        ref.fabricate()
        ref.snapshot()
    assert p.energy_events == []
    assert p.latency_events == []


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_toml_array_loads_as_tuple(case: _Case, tmp_path: Path) -> None:
    """A TOML array loads straight into the tuple field (nested for the 2-D bank)."""
    toml = (
        "[ref]\n"
        f"{case.field} = {case.toml_array}\n"
        "tolerance_sigma_relative = 0.01\n"
        "noise_sigma_relative = 0.002\n"
        "area_per_inst__um2 = 1.0\n"
        "leakage_per_inst__uW = 0.5\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")

    config = case.config_cls.from_file(path, section="ref")
    loaded = getattr(config, case.field)
    assert isinstance(loaded, tuple)
    assert loaded == case.taps


def test_current_toml_flat_array_rejected(tmp_path: Path) -> None:
    """A stale flat TOML array does not deserialize into the 2-D bank field."""
    toml = (
        "[ref]\n"
        "i_refs__uA = [1.0, 5.0, 20.0]\n"
        "tolerance_sigma_relative = 0.0\n"
        "noise_sigma_relative = 0.0\n"
        "area_per_inst__um2 = 0.0\n"
        "leakage_per_inst__uW = 0.0\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(TypeError):
        CurrentReferenceConfig.from_file(path, section="ref")
