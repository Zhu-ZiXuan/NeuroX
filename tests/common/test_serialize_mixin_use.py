"""Configuration file sections, path conversion, and YAML value normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from neurox.common.serialize import dict_from_file
from neurox.common.serialize_mixin import SerializeMixin


@dataclass(frozen=True)
class _Point(SerializeMixin):
    x: float
    y: float


@dataclass(frozen=True)
class _PathBox(SerializeMixin):
    path: Path


# --- file round-trip (whole file) ---


def test_to_file_from_file_round_trip(tmp_path: Path) -> None:
    point = _Point(x=3.0, y=4.0)
    file = tmp_path / "point.toml"
    point.to_file(file)
    loaded = _Point.from_file(file)
    assert loaded == point


# --- file round-trip (section) ---


def test_to_file_from_file_section_round_trip(tmp_path: Path) -> None:
    point = _Point(x=5.0, y=6.0)
    file = tmp_path / "point.toml"
    point.to_file(file, section="foo")
    loaded = _Point.from_file(file, section="foo")
    assert loaded == point
    assert "[foo]" in file.read_text()


def test_path_round_trip_uses_string_value(tmp_path: Path) -> None:
    expected = _PathBox(path=Path("models/config.toml"))
    assert expected.to_dict() == {"path": "models/config.toml"}

    file = tmp_path / "path.toml"
    expected.to_file(file)
    assert _PathBox.from_file(file) == expected


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("1: value\n", "configuration mapping key must be str"),
        ("value: 2025-01-01\n", "unsupported configuration value date"),
    ],
)
def test_yaml_loader_rejects_values_outside_config_contract(tmp_path: Path, body: str, message: str) -> None:
    file = tmp_path / "invalid.yaml"
    file.write_text(body)
    with pytest.raises(TypeError, match=message):
        dict_from_file(file)
