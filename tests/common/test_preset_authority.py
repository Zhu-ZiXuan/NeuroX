"""Preset-authority semantics.

`from_preset` dispatch, receiver-bounded resolution, and directive-versus-`_neurox_class` exclusion.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import pytest

from neurox.common import SerializeMixin
from neurox.primitive.device import MosfetConfig, RramConfig

# --- 1. from_preset happy path: bundled preset -> the receiver's own type ---


def test_from_preset_rram_builds_expected_instance() -> None:
    cfg = RramConfig.from_preset("process/rram:default")
    assert isinstance(cfg, RramConfig)
    assert cfg.g_min__uS == 10.0
    assert cfg.nonlinearity_alpha == 0.5


def test_from_preset_mosfet_builds_expected_instance() -> None:
    cfg = MosfetConfig.from_preset("process/mos:nmos_28_rvt")
    assert isinstance(cfg, MosfetConfig)
    assert cfg.vth0__V == 0.40
    assert cfg.n_factor == 1.25


# --- 2. subtype guard: a preset for a foreign class is rejected by the receiver ---


def test_from_preset_wrong_receiver_raises_type_mismatch() -> None:
    with pytest.raises(TypeError) as exc:
        MosfetConfig.from_preset("process/rram:default")
    msg = str(exc.value)
    assert "RramConfig" in msg
    assert "MosfetConfig" in msg


# --- 3. rule #1: a directive and _neurox_class may not co-occur in one table ---


def test_use_preset_and_class_discriminator_conflict(tmp_path: Path) -> None:
    file = tmp_path / "conflict_preset.toml"
    file.write_text(
        '[thing]\n_neurox_use_preset = "process/rram:default"\n_neurox_class = "RramConfig"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'_neurox_use_preset' table may not also declare '_neurox_class'"):
        RramConfig.from_file(file, section="thing")


def test_use_and_class_discriminator_conflict(tmp_path: Path) -> None:
    file = tmp_path / "conflict_use.toml"
    # The conflict fires before path resolution, so the fragment need not exist.
    file.write_text(
        '[thing]\n_neurox_use = "fragment:sec"\n_neurox_class = "RramConfig"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'_neurox_use' table may not also declare '_neurox_class'"):
        RramConfig.from_file(file, section="thing")


# --- 4. inline-table idiom == section-header idiom == direct from_preset ---


@dataclass(frozen=True)
class _RramBox(SerializeMixin):
    """Container with a single polymorphic-slot field typed as `RramConfig`."""

    device: RramConfig


def test_inline_table_matches_section_header_and_direct_preset(tmp_path: Path) -> None:
    inline = tmp_path / "inline.toml"
    inline.write_text(
        'device = { _neurox_use_preset = "process/rram:default" }\n',
        encoding="utf-8",
    )
    section = tmp_path / "section.toml"
    section.write_text(
        '[device]\n_neurox_use_preset = "process/rram:default"\n',
        encoding="utf-8",
    )

    box_inline = _RramBox.from_file(inline)
    box_section = _RramBox.from_file(section)
    direct = RramConfig.from_preset("process/rram:default")

    assert type(box_inline.device) is RramConfig
    assert box_inline.device == direct
    assert box_inline == box_section


# --- 5. abstract-base rule still fires after the coerce/directive split ---


@dataclass(frozen=True)
class _Base(SerializeMixin, ABC):
    """Polymorphic base: abstract via the declared `ABC` signal."""


@dataclass(frozen=True)
class _Leaf(_Base):
    pass


def test_abstract_base_from_dict_raises() -> None:
    with pytest.raises(TypeError) as exc:
        _Base.from_dict({})
    msg = str(exc.value)
    assert "abstract" in msg
    assert "subclass" in msg


# --- 6. self-describing leaf: _neurox_class naming the receiver itself resolves ---


def test_self_describing_leaf_resolves_to_receiver() -> None:
    expected = RramConfig.from_preset("process/rram:default")
    # A leaf preset carries _neurox_class equal to its own class name; the
    # resolver must consider the receiver ("base itself"), not only subclasses.
    data = {**expected.to_dict(), "_neurox_class": "RramConfig"}
    resolved = RramConfig.from_dict(data)
    assert type(resolved) is RramConfig
    assert resolved == expected
