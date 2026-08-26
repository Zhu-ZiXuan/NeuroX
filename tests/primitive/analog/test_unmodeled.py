"""UnmodeledBlock: lumped PPA for a functionally unmodeled block.

The block carries area, leakage, and flat per-operation energy. Static PPA
scales by `inst_count`; `execute` records one event at every requested shape
position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog import UnmodeledBlock, UnmodeledBlockConfig, UnmodeledBlockPolicy


def _config(**overrides: Any) -> UnmodeledBlockConfig:
    base = {"area_per_inst__um2": 4.0, "leakage_per_inst__uW": 0.5, "energy_per_op__fJ": 2.0}
    return UnmodeledBlockConfig(**{**base, **overrides})


def _make(*, area: float = 4.0, leakage: float = 0.5, inst_shape: tuple[int, ...] = ()) -> UnmodeledBlock:
    return UnmodeledBlock(
        config=_config(area_per_inst__um2=area, leakage_per_inst__uW=leakage),
        policy=UnmodeledBlockPolicy(),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )


def test_validation_rejects_negative_ppa() -> None:
    """Negative PPA is rejected; zero is allowed."""
    for override, match in (
        ({"area_per_inst__um2": -1e-3}, r"require: area_per_inst__um2 \(-0\.001\) >= 0"),
        ({"leakage_per_inst__uW": -1e-3}, r"require: leakage_per_inst__uW \(-0\.001\) >= 0"),
        ({"energy_per_op__fJ": -1e-3}, r"require: energy_per_op__fJ \(-0\.001\) >= 0"),
    ):
        with pytest.raises(ValueError, match=match):
            _config(**override)
    _config(area_per_inst__um2=0.0, leakage_per_inst__uW=0.0, energy_per_op__fJ=0.0)


def test_static_ppa_scales_and_is_visible() -> None:
    """Static PPA scales by `inst_count` and appears in the profiler's static walk."""
    block = _make(area=4.0, leakage=0.5, inst_shape=(2,))
    stamp_names(block)
    assert block.is_profile_target
    assert block.area__um2 == pytest.approx(4.0 * 2)
    assert block.leakage__uW == pytest.approx(0.5 * 2)

    entries = Reporter(block).static_entries
    assert len(entries) == 1
    assert entries[0].area__um2 == pytest.approx(4.0 * 2)
    assert entries[0].leakage__uW == pytest.approx(0.5 * 2)


def test_fabricate_emits_no_events() -> None:
    """The block has no functional path; fabricate emits no energy record."""
    block = _make(inst_shape=(2,))
    with Profiler() as p:
        block.fabricate()
    assert p.records == ()


def test_execute_records_one_operation_per_shape_position() -> None:
    block = _make(inst_shape=(2,))
    stamp_names(block)
    with Profiler() as profiler:
        block.execute((3, 2))
    assert Reporter(block).total_dynamic_energy__fJ(profiler) == pytest.approx(2.0 * 3 * 2)


def test_toml_loads(tmp_path: Path) -> None:
    """A TOML section loads straight into the PPA fields."""
    toml = "[block]\narea_per_inst__um2 = 4.0\nleakage_per_inst__uW = 0.5\nenergy_per_op__fJ = 2.0\n"
    path = tmp_path / "block.toml"
    path.write_text(toml, encoding="utf-8")

    config = UnmodeledBlockConfig.from_file(path, section="block")
    assert config.area_per_inst__um2 == pytest.approx(4.0)
    assert config.leakage_per_inst__uW == pytest.approx(0.5)
    assert config.energy_per_op__fJ == pytest.approx(2.0)
