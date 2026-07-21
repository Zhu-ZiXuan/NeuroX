"""CurrentReference: 2-D mode/tap bank and the PPA-only contract.

``CurrentReference`` is a behavioural reference source: it carries static
PPA (area + leakage) and hands out the actual tap values through a snap,
but performs no computation and emits no dynamic energy or latency. It
holds a 2-D ``[mode][tap]`` bank of strictly increasing equal-length rows
whose mode selection is quasi-static. These tests pin, all with the policy
all-off:

- 2-D validation on the bank: strictly increasing rows, equal row
  lengths, and non-negative taps;
- ``mode_num`` / ``tap_num`` report the bank geometry;
- ``snapshot`` is deterministic and matches the nominal taps, broadcast
  to ``(*inst_shape, mode_num, tap_num)``;
- the ``i_ref__uA`` accessor returns every tap from a snap (the
  encapsulated read path consumers use instead of the buffer);
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` + ``snapshot`` emit zero
  energy / latency events;
- a nested TOML array loads straight into the bank field; a flat TOML
  array (wrong shape for the tap bank) is rejected.
"""

from __future__ import annotations

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

_TAPS = ((1.0, 5.0, 20.0), (2.0, 6.0, 25.0))
_BANK_SHAPE = (2, 3)


def _config(**overrides: Any) -> CurrentReferenceConfig:
    base = {
        "i_refs__uA": _TAPS,
        "tolerance_sigma_relative": 0.0,
        "noise_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return CurrentReferenceConfig(**{**base, **overrides})


def _make(
    *,
    inst_shape: tuple[int, ...] = (),
    area: float = 0.0,
    leakage: float = 0.0,
) -> CurrentReference:
    return CurrentReference(
        config=_config(area_per_inst__um2=area, leakage_per_inst__uW=leakage),
        policy=CurrentReferencePolicy(tolerance=False, noise=False),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


def test_validation_rejects_bad_config() -> None:
    """Empty taps and negative sigmas/PPA are rejected."""
    for override in (
        {"i_refs__uA": ()},  # empty tap list
        {"tolerance_sigma_relative": -1e-3},
        {"noise_sigma_relative": -1e-3},
        {"area_per_inst__um2": -1.0},
        {"leakage_per_inst__uW": -1.0},
    ):
        with pytest.raises(ValueError):
            _config(**override)


def test_bank_2d_validation() -> None:
    """The bank enforces strictly increasing, equal-length, non-negative rows."""
    for bad in (
        ((0.6, -1.0),),  # negative tap (also non-increasing)
        ((2.0, 1.0, 3.0),),  # non-increasing row
        ((1.0, 1.0),),  # not strictly increasing
        ((1.0, 2.0), (1.0, 2.0, 3.0)),  # unequal row lengths
        ((),),  # empty row
    ):
        with pytest.raises(ValueError):
            _config(i_refs__uA=bad)
    # A 0 first tap denotes a ground/rail reference and is accepted.
    _config(i_refs__uA=((0.0, 0.6),))


def test_mode_tap_counts() -> None:
    """``mode_num`` / ``tap_num`` report the bank geometry on config and module."""
    ref = _make()
    assert ref.config.mode_num == 2
    assert ref.config.tap_num == 3
    assert ref.mode_num == 2
    assert ref.tap_num == 3


def test_all_off_snapshot_matches_nominal() -> None:
    """All-off ``snapshot`` is the deterministic nominal bank."""
    ref = _make()
    nominal = torch.tensor(_TAPS, dtype=torch.float64)

    snap_a = ref.snapshot()
    snap_b = ref.snapshot()
    assert snap_a.i_refs__uA.shape == _BANK_SHAPE
    torch.testing.assert_close(snap_a.i_refs__uA, nominal)
    torch.testing.assert_close(snap_b.i_refs__uA, nominal)


def test_inst_shape_broadcasts_taps() -> None:
    """A non-scalar ``inst_shape`` yields ``(*inst_shape, mode_num, tap_num)`` taps."""
    ref = _make(inst_shape=(1, 2))
    out = ref.snapshot().i_refs__uA
    nominal = torch.tensor(_TAPS, dtype=torch.float64)
    assert out.shape == (1, 2, *_BANK_SHAPE)
    torch.testing.assert_close(out, nominal.expand(1, 2, *_BANK_SHAPE))


def test_accessor_reads_taps_from_snap() -> None:
    """The ``i_ref__uA`` accessor returns every tap from a snap."""
    ref = _make(inst_shape=(1, 2))
    ref.fabricate()
    snap = ref.snapshot()
    out = ref.i_ref__uA(snap)

    # Encapsulated read: returns the snap's own tensor, full tap set.
    assert out.shape == (1, 2, *_BANK_SHAPE)
    assert torch.equal(out, snap.i_refs__uA)
    # Pure: re-reading the same snap returns the same tensor.
    assert torch.equal(ref.i_ref__uA(snap), out)


def test_static_ppa_and_no_dynamic_events() -> None:
    """Static PPA scales by ``inst_count``; fabricate/snapshot emit no events."""
    ref = _make(inst_shape=(2,), area=2.0, leakage=0.5)

    assert ref.area__um2 == pytest.approx(2.0 * 2)
    assert ref.leakage__uW == pytest.approx(0.5 * 2)

    records = NeuroxProfiler.collect_static(ref)
    assert len(records) == 1
    assert records[0].area__um2 == pytest.approx(2.0 * 2)
    assert records[0].leakage_power__uW == pytest.approx(0.5 * 2)

    with NeuroxProfiler() as p:
        ref.fabricate()
        ref.snapshot()
    assert p.energy_events == []
    assert p.latency_events == []


def test_toml_nested_array_loads_as_tuple(tmp_path: Path) -> None:
    """A nested TOML array loads straight into the 2-D bank field."""
    toml = (
        "[ref]\n"
        "i_refs__uA = [[1.0, 5.0, 20.0], [2.0, 6.0, 25.0]]\n"
        "tolerance_sigma_relative = 0.01\n"
        "noise_sigma_relative = 0.002\n"
        "area_per_inst__um2 = 1.0\n"
        "leakage_per_inst__uW = 0.5\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")

    config = CurrentReferenceConfig.from_file(path, section="ref")
    assert isinstance(config.i_refs__uA, tuple)
    assert config.i_refs__uA == _TAPS


def test_toml_flat_array_rejected(tmp_path: Path) -> None:
    """A flat TOML array (wrong shape for the tap bank) does not deserialize into the 2-D bank field."""
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
