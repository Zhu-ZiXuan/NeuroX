"""Dict <-> file I/O for TOML and YAML."""

import tomllib
from collections.abc import Mapping
from pathlib import Path

import tomli_w
import yaml

from .value import ConfigDict, ConfigValue, normalize_config_dict


def dict_from_toml(file: Path) -> ConfigDict:
    """Load a dict from a TOML file."""
    with file.open(mode="rb") as f:
        data: object = tomllib.load(f)
    return normalize_config_dict(data)


def dict_to_toml(data: Mapping[str, ConfigValue], file: Path) -> None:
    """Write a mapping to a TOML file.

    TOML has no null literal, so a `None` value is dropped from a mapping and
    its key is absent from the file. A `None` inside a list is not dropped and
    raises `TypeError`.
    """
    normalized = normalize_config_dict(data)
    with file.open(mode="wb") as f:
        tomli_w.dump(_strip_none_dict(normalized), f)


def _strip_none_dict(data: ConfigDict) -> ConfigDict:
    """Recursively drop `None` values from a configuration mapping."""
    return {key: _strip_none_value(value) for key, value in data.items() if value is not None}


def _strip_none_value(data: ConfigValue) -> ConfigValue:
    """Recursively drop `None` values from any mapping one value holds."""
    if isinstance(data, dict):
        return _strip_none_dict(data)
    if isinstance(data, list):
        return [_strip_none_value(x) for x in data]
    return data


def dict_from_yaml(file: Path, *, encoding: str | None = "utf-8") -> ConfigDict:
    """Load a dict from a YAML file."""
    with file.open(mode="r", encoding=encoding) as f:
        data: object = yaml.safe_load(f)
    return normalize_config_dict(data)


def dict_to_yaml(data: Mapping[str, ConfigValue], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Write a mapping to a YAML file."""
    normalized = normalize_config_dict(data)
    with file.open(mode="w", encoding=encoding) as f:
        yaml.safe_dump(
            data=normalized,
            stream=f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


toml_suffixes = {".toml"}
yaml_suffixes = {".yaml", ".yml"}
supported_suffixes = toml_suffixes | yaml_suffixes


def dict_from_file(file: Path, *, encoding: str | None = "utf-8") -> ConfigDict:
    """Load a dict from a TOML or YAML file (dispatched by suffix).

    `encoding` is the YAML text encoding and is ignored for TOML.
    """
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        return dict_from_toml(file)
    if suffix in yaml_suffixes:
        return dict_from_yaml(file, encoding=encoding)
    raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")


def dict_to_file(data: Mapping[str, ConfigValue], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Write a mapping to a TOML or YAML file (dispatched by suffix)."""
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        dict_to_toml(data, file)
    elif suffix in yaml_suffixes:
        dict_to_yaml(data, file, encoding=encoding)
    else:
        raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")
