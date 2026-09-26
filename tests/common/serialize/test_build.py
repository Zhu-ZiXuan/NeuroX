"""Nested polymorphic construction respects receiver families and invokes validation hooks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import pytest

from neurox.common.module import ConfigBase, PolicyBase
from neurox.common.serialize import ConfigDict
from neurox.common.serialize_mixin import SerializeMixin


class _ConfigFamily(ConfigBase, base_only=True):
    pass


class _Config(_ConfigFamily):
    value: int

    def validate(self) -> None:
        self._require_non_neg(self.value, "value")


class _PolicyFamily(PolicyBase, base_only=True):
    pass


class _Policy(_PolicyFamily):
    count: int

    def validate(self) -> None:
        self._require_non_neg(self.count, "count")


@dataclass(frozen=True)
class _Run(SerializeMixin):
    configuration: _ConfigFamily
    policy: _PolicyFamily


@dataclass(frozen=True)
class _ConcreteRoot(SerializeMixin):
    pass


@dataclass(frozen=True)
class _Skin(_ConcreteRoot):
    pass


type _FloatArray = float | tuple[_FloatArray, ...]


@dataclass(frozen=True)
class _ArrayConfig(SerializeMixin):
    values: _FloatArray


def _data(*, value: int, count: int) -> ConfigDict:
    return {
        "configuration": {"_neurox_class": "_Config", "value": value},
        "policy": {"_neurox_class": "_Policy", "count": count},
    }


def test_nested_discriminators_resolve_only_within_the_declared_family() -> None:
    data = _data(value=2, count=3)
    run = _Run.from_dict(data)
    assert isinstance(run.configuration, _Config)
    assert isinstance(run.policy, _Policy)
    assert run.to_dict() == data

    data["configuration"] = {"_neurox_class": "_Policy", "count": 3}
    with pytest.raises(TypeError):
        _Run.from_dict(data)


@pytest.mark.parametrize(
    ("value", "count", "invalid_field"), [(-1, 3, "value"), (2, -1, "count")], ids=["config", "policy"]
)
def test_loading_nested_types_runs_their_inherited_validation_hook(value: int, count: int, invalid_field: str) -> None:
    with pytest.raises(ValueError, match=invalid_field):
        _Run.from_dict(_data(value=value, count=count))


def test_a_concrete_type_with_descendants_can_still_be_loaded_directly() -> None:
    assert type(_ConcreteRoot.from_dict({})) is _ConcreteRoot


def test_recursive_type_alias_coerces_nested_toml_values(tmp_path: Path) -> None:
    path = tmp_path / "array.toml"
    path.write_text("values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]\n", encoding="utf-8")
    config = _ArrayConfig.from_file(path)
    assert config.values == (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))


class _Mode(Enum):
    LOW = "low"


@dataclass(frozen=True)
class _LiteralConfig(SerializeMixin):
    code: Literal[0, 1]
    enabled: Literal[True]
    mode: Literal[_Mode.LOW]


def test_literal_fields_round_trip_without_erasing_types() -> None:
    data = {"code": 1, "enabled": True, "mode": "low"}
    config = _LiteralConfig.from_dict(data)
    assert type(config.code) is int
    assert config.enabled is True
    assert config.mode is _Mode.LOW
    assert config.to_dict() == data


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [("code", True, TypeError), ("code", 1.0, TypeError), ("enabled", 1, TypeError), ("code", 2, ValueError)],
)
def test_literal_fields_distinguish_type_mismatches_from_invalid_choices(field, value, error) -> None:
    data = {"code": 1, "enabled": True, "mode": "low"}
    data[field] = value
    with pytest.raises(error, match=field):
        _LiteralConfig.from_dict(data)
