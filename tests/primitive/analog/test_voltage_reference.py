"""VoltageReference: flat multi-tap source and the PPA-only contract.

``VoltageReference`` is a behavioural reference source: it carries static
PPA (area + leakage) and hands out the actual tap values through a snap,
but performs no computation and emits no dynamic energy or latency. It
holds a flat unordered tap tuple. These tests pin, all with the policy
all-off:

- config validation (empty / negative taps, negative sigmas / PPA);
- ``snapshot`` is deterministic and matches the nominal taps, broadcast
  to ``(*inst_shape, tap_num)``;
- the ``v_ref__V`` accessor returns every tap from a snap (the
  encapsulated read path consumers use instead of the buffer);
- static PPA equals ``per_inst * inst_count`` and is visible to the
  profiler's static walk, while ``fabricate`` + ``snapshot`` emit zero
  energy / latency events;
- a TOML array loads straight into the tuple field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.voltage_reference import (
    VoltageReference,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
)

_TAPS = (0.6, 1.2, 0.3)


def _config(**overrides: Any) -> VoltageReferenceConfig:
    base = {
        "v_refs__V": _TAPS,
        "tolerance_sigma_relative": 0.0,
        "noise_sigma_relative": 0.0,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
    }
    return VoltageReferenceConfig(**{**base, **overrides})


def _make(
    *,
    inst_shape: tuple[int, ...] = (),
    area: float = 0.0,
    leakage: float = 0.0,
) -> VoltageReference:
    return VoltageReference(
        config=_config(area_per_inst__um2=area, leakage_per_inst__uW=leakage),
        policy=VoltageReferencePolicy(tolerance=False, noise=False),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


def test_validation_rejects_bad_config() -> None:
    """Empty taps, negative taps, and negative sigmas/PPA are rejected."""
    for override in (
        {"v_refs__V": ()},  # empty tap list
        {"v_refs__V": (0.6, -1.0)},  # negative tap
        {"tolerance_sigma_relative": -1e-3},
        {"noise_sigma_relative": -1e-3},
        {"area_per_inst__um2": -1.0},
        {"leakage_per_inst__uW": -1.0},
    ):
        with pytest.raises(ValueError):
            _config(**override)
    # Taps stay a flat unordered tuple; a 0 tap (ground rail) is accepted.
    _config(v_refs__V=(0.6, 0.0))


def test_all_off_snapshot_matches_nominal() -> None:
    """All-off ``snapshot`` is the deterministic nominal taps."""
    ref = _make()
    nominal = torch.tensor(_TAPS, dtype=torch.float64)

    snap_a = ref.snapshot()
    snap_b = ref.snapshot()
    assert snap_a.v_refs__V.shape == (len(_TAPS),)
    torch.testing.assert_close(snap_a.v_refs__V, nominal)
    torch.testing.assert_close(snap_b.v_refs__V, nominal)


def test_inst_shape_broadcasts_taps() -> None:
    """A non-scalar ``inst_shape`` yields ``(*inst_shape, tap_num)`` taps."""
    ref = _make(inst_shape=(1, 2))
    out = ref.snapshot().v_refs__V
    nominal = torch.tensor(_TAPS, dtype=torch.float64)
    assert out.shape == (1, 2, len(_TAPS))
    torch.testing.assert_close(out, nominal.expand(1, 2, len(_TAPS)))


def test_accessor_reads_taps_from_snap() -> None:
    """The ``v_ref__V`` accessor returns every tap from a snap."""
    ref = _make(inst_shape=(1, 2))
    ref.fabricate()
    snap = ref.snapshot()
    out = ref.v_ref__V(snap)

    # Encapsulated read: returns the snap's own tensor, full tap set.
    assert out.shape == (1, 2, len(_TAPS))
    assert torch.equal(out, snap.v_refs__V)
    # Pure: re-reading the same snap returns the same tensor.
    assert torch.equal(ref.v_ref__V(snap), out)


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


def test_toml_array_loads_as_tuple(tmp_path: Path) -> None:
    """A TOML array loads straight into the flat tuple field."""
    toml = (
        "[ref]\n"
        "v_refs__V = [0.6, 1.2, 0.3]\n"
        "tolerance_sigma_relative = 0.01\n"
        "noise_sigma_relative = 0.002\n"
        "area_per_inst__um2 = 1.0\n"
        "leakage_per_inst__uW = 0.5\n"
    )
    path = tmp_path / "ref.toml"
    path.write_text(toml, encoding="utf-8")

    config = VoltageReferenceConfig.from_file(path, section="ref")
    assert isinstance(config.v_refs__V, tuple)
    assert config.v_refs__V == _TAPS
