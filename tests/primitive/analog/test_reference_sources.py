"""Reference sources: multi-tap state, tolerance/noise, and the PPA-only contract.

``VoltageReference`` / ``CurrentReference`` are behavioural reference
sources: they carry static PPA (area + leakage) and hand out the actual
tap values through a snap, but perform no computation and emit no dynamic
energy or latency. These tests pin:

- multi-tap config loads a TOML array straight into the ``tuple`` field;
- ``snapshot`` is deterministic with both nonidealities off and matches
  the nominal taps, broadcast to ``(*inst_shape, num_refs)``;
- the ``v_ref__V`` / ``i_ref__uA`` accessor returns every tap from a
  snap (the encapsulated read path consumers use instead of the buffer);
- ``tolerance`` is a per-die static spread fixed at ``fabricate`` time;
  ``noise`` is resampled every ``snapshot``;
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` + ``snapshot`` emit zero
  energy / latency events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

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
from neurox.common.load_dump import dataclass_from_file
from neurox.common.profiler import NeuroxProfiler

# (ref_cls, config_cls, policy_cls, field_name, nominal_taps)
_CASES = [
    (VoltageReference, VoltageReferenceConfig, VoltageReferencePolicy, "v_refs__V", (0.6, 1.2, 0.3)),
    (CurrentReference, CurrentReferenceConfig, CurrentReferencePolicy, "i_refs__uA", (1.0, 5.0, 20.0)),
]
_IDS = ["voltage", "current"]
# field -> the encapsulated snap-read accessor method name.
_ACCESSOR = {"v_refs__V": "v_ref__V", "i_refs__uA": "i_ref__uA"}


def _make(
    case: tuple[Any, Any, Any, str, tuple[float, ...]],
    *,
    inst_shape: tuple[int, ...] = (),
    tolerance: bool = False,
    noise: bool = False,
    tol_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    area: float = 0.0,
    leakage: float = 0.0,
) -> Any:
    ref_cls, config_cls, policy_cls, field, taps = case
    config = config_cls(
        **{
            field: taps,
            "tolerance_sigma_relative": tol_sigma,
            "noise_sigma_relative": noise_sigma,
            "area_per_inst__um2": area,
            "leakage_per_inst__uW": leakage,
        }
    )
    return ref_cls(
        config=config,
        policy=policy_cls(tolerance=tolerance, noise=noise),
        name="ref",
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_validation_rejects_bad_config(case: tuple[Any, ...]) -> None:
    """Empty taps, negative taps, and negative sigmas/PPA are rejected; a 0 tap is accepted."""
    _, config_cls, _, field, taps = case
    base = {
        field: taps,
        "tolerance_sigma_relative": 0.0,
        "noise_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    for override in (
        {field: ()},  # empty tap list
        {field: (0.6, -1.0)},  # negative tap
        {"tolerance_sigma_relative": -1e-3},
        {"noise_sigma_relative": -1e-3},
        {"area_per_inst__um2": -1.0},
        {"leakage_per_inst__uW": -1.0},
    ):
        with pytest.raises(ValueError):
            config_cls(**{**base, **override})

    # A 0 tap denotes a ground/rail reference and is accepted.
    config_cls(**{**base, field: (0.6, 0.0)})


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_all_off_snapshot_matches_nominal(case: tuple[Any, ...]) -> None:
    """With both nonidealities off, ``snapshot`` is the deterministic nominal taps."""
    *_, field, taps = case
    ref = _make(case)
    assert ref.num_refs == len(taps)

    snap_a = ref.snapshot()
    snap_b = ref.snapshot()
    out_a = getattr(snap_a, field)
    nominal = torch.tensor(taps, dtype=torch.float64)

    assert out_a.shape == (len(taps),)
    torch.testing.assert_close(out_a, nominal)
    torch.testing.assert_close(getattr(snap_b, field), nominal)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_inst_shape_broadcasts_taps(case: tuple[Any, ...]) -> None:
    """A non-scalar ``inst_shape`` yields ``(*inst_shape, num_refs)`` taps."""
    *_, field, taps = case
    ref = _make(case, inst_shape=(2, 3))
    out = getattr(ref.snapshot(), field)
    nominal = torch.tensor(taps, dtype=torch.float64)
    assert out.shape == (2, 3, len(taps))
    torch.testing.assert_close(out, nominal.expand(2, 3, len(taps)))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_accessor_reads_taps_from_snap(case: tuple[Any, ...]) -> None:
    """The ``v_ref__V`` / ``i_ref__uA`` accessor returns every tap from a snap."""
    *_, field, _taps = case
    ref = _make(case, inst_shape=(2, 3), noise=True, noise_sigma=0.05)
    ref.fabricate()
    accessor = getattr(ref, _ACCESSOR[field])

    torch.manual_seed(0)
    snap = ref.snapshot()
    out = accessor(snap)

    # Encapsulated read: returns the snap's own (noised) tensor, full tap set.
    assert out.shape == (2, 3, ref.num_refs)
    assert torch.equal(out, getattr(snap, field))
    # Pure: re-reading the same snap never resamples.
    assert torch.equal(accessor(snap), out)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_tolerance_is_static_per_die(case: tuple[Any, ...]) -> None:
    """``tolerance`` is fixed at fabricate time and resampled only by ``fabricate``."""
    *_, field, taps = case
    nominal = torch.tensor(taps, dtype=torch.float64)
    ref = _make(case, inst_shape=(8,), tolerance=True, tol_sigma=0.1)

    # Before fabricate(): the actual buffer is the untouched nominal.
    torch.testing.assert_close(getattr(ref, field), nominal.expand(8, len(taps)))

    torch.manual_seed(0)
    ref.fabricate()
    fab1 = getattr(ref, field).clone()
    assert fab1.shape == (8, len(taps))
    assert not torch.allclose(fab1, nominal.expand(8, len(taps)))

    # noise is off, so repeated snapshots return the same per-die taps.
    torch.testing.assert_close(getattr(ref.snapshot(), field), fab1)
    torch.testing.assert_close(getattr(ref.snapshot(), field), fab1)

    # A second fabricate() resamples the static spread.
    torch.manual_seed(1)
    ref.fabricate()
    assert not torch.allclose(getattr(ref, field), fab1)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_noise_resamples_every_snapshot(case: tuple[Any, ...]) -> None:
    """``noise`` makes consecutive snapshots differ."""
    *_, field, _taps = case
    ref = _make(case, inst_shape=(4,), noise=True, noise_sigma=0.05)
    torch.manual_seed(0)
    first = getattr(ref.snapshot(), field)
    second = getattr(ref.snapshot(), field)
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_static_ppa_and_no_dynamic_events(case: tuple[Any, ...]) -> None:
    """Static PPA scales by ``inst_count``; fabricate/snapshot emit no events."""
    ref = _make(
        case, inst_shape=(3,), tolerance=True, noise=True, tol_sigma=0.1, noise_sigma=0.05, area=2.0, leakage=0.5
    )

    assert ref.inst_area__um2 == pytest.approx(2.0 * 3)
    assert ref.inst_leakage__uW == pytest.approx(0.5 * 3)

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
def test_toml_array_loads_as_tuple(case: tuple[Any, ...], tmp_path: Path) -> None:
    """A TOML array (list) loads straight into the ``tuple[float, ...]`` field."""
    _, config_cls, _, field, taps = case
    array = ", ".join(repr(v) for v in taps)
    toml = (
        "[ref]\n"
        f"{field} = [{array}]\n"
        "tolerance_sigma_relative = 0.01\n"
        "noise_sigma_relative = 0.002\n"
        "area_per_inst__um2 = 1.0\n"
        "leakage_per_inst__uW = 0.5\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")

    config = dataclass_from_file(config_cls, path, section="ref")
    loaded = getattr(config, field)
    assert isinstance(loaded, tuple)
    assert loaded == taps
    assert all(isinstance(v, float) for v in loaded)
