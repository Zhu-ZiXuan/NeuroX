"""Nested polymorphic construction respects receiver families and invokes validation hooks."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import pytest

from neurox.common.module import ConfigBase, PolicyBase
from neurox.common.serialize import ConfigDict
from neurox.common.serialize_mixin import SerializeMixin


class _ConfigFamily(ConfigBase, ABC):
    pass


class _Config(_ConfigFamily):
    value: int

    def validate(self) -> None:
        self._require_non_neg(self.value, "value")


class _PolicyFamily(PolicyBase, ABC):
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
