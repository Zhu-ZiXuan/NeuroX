"""Canonical values exchanged by NeuroX configuration serializers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

ConfigScalar: TypeAlias = None | bool | int | float | str
ConfigValue: TypeAlias = ConfigScalar | list["ConfigValue"] | dict[str, "ConfigValue"]
ConfigDict: TypeAlias = dict[str, ConfigValue]


def normalize_config_value(value: object, *, path: str = "<root>") -> ConfigValue:
    """Validate and copy one value into the canonical configuration tree."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [normalize_config_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        normalized: ConfigDict = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: configuration mapping key must be str, got {type(key).__name__}")
            normalized[key] = normalize_config_value(item, path=f"{path}.{key}")
        return normalized
    raise TypeError(f"{path}: unsupported configuration value {type(value).__name__}")


def normalize_config_dict(value: object, *, path: str = "<root>") -> ConfigDict:
    """Validate and copy a canonical configuration mapping."""
    normalized = normalize_config_value(value, path=path)
    if not isinstance(normalized, dict):
        raise TypeError(f"{path}: configuration root must be a mapping, got {type(normalized).__name__}")
    return normalized
