"""Tests for the ``SerializeMixin`` {dict, file} x {read, write} surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from neurox.common import SerializeMixin
from neurox.common.serialize import dict_from_file


@dataclass(frozen=True)
class _Point(SerializeMixin):
    x: float
    y: float


@dataclass(frozen=True)
class _Shape(SerializeMixin):
    """Base of a small polymorphic family for the discriminator test."""

    name: str


@dataclass(frozen=True)
class _Circle(_Shape):
    radius: float


@dataclass(frozen=True)
class _PathBox(SerializeMixin):
    path: Path


@dataclass(frozen=True)
class _UnsupportedValue(SerializeMixin):
    value: complex


# --- dict round-trip ---


def test_from_dict_to_dict_round_trip() -> None:
    data = {"x": 1.5, "y": -2.0}
    point = _Point.from_dict(data)
    assert point == _Point(x=1.5, y=-2.0)
    assert point.to_dict() == data


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


# --- polymorphic round-trip ---


def test_from_dict_discriminator_yields_subclass_instance() -> None:
    data = {"_neurox_class": "_Circle", "name": "c1", "radius": 2.5}
    shape = _Shape.from_dict(data)
    assert isinstance(shape, _Circle)
    assert shape == _Circle(name="c1", radius=2.5)


def test_path_round_trip_uses_string_value(tmp_path: Path) -> None:
    expected = _PathBox(path=Path("models/config.toml"))
    assert expected.to_dict() == {"path": "models/config.toml"}

    file = tmp_path / "path.toml"
    expected.to_file(file)
    assert _PathBox.from_file(file) == expected


def test_to_dict_rejects_unsupported_leaf_value() -> None:
    with pytest.raises(TypeError, match="unsupported configuration value complex"):
        _UnsupportedValue(value=1 + 2j).to_dict()


@pytest.mark.parametrize(
    "body, message",
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
