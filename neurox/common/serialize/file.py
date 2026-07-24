"""Dict <-> file I/O for TOML and YAML."""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w
import yaml


def dict_from_toml(file: Path) -> dict[str, Any]:
    """Load a dict from a TOML file."""
    with file.open(mode="rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"TOML root must be a table, got {type(data).__name__}")
    return data


def dict_to_toml(data: Mapping[str, Any], file: Path) -> None:
    """Write a mapping to a TOML file.

    TOML has no null literal, so a ``None`` value is dropped from a mapping and
    its key is absent from the file. A ``None`` inside a list is not dropped and
    raises ``TypeError``.
    """
    with file.open(mode="wb") as f:
        tomli_w.dump(_strip_none(data), f)


def _strip_none(data: Any) -> Any:
    """Recursively drop ``None`` values from mappings."""
    if isinstance(data, Mapping):
        return {k: _strip_none(v) for k, v in data.items() if v is not None}
    if isinstance(data, list | tuple):
        return [_strip_none(x) for x in data]
    return data


def dict_from_yaml(file: Path, *, encoding: str | None = "utf-8") -> dict[str, Any]:
    """Load a dict from a YAML file."""
    with file.open(mode="r", encoding=encoding) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(data).__name__}")
    return data


def dict_to_yaml(data: Mapping[str, Any], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Write a mapping to a YAML file."""
    with file.open(mode="w", encoding=encoding) as f:
        yaml.dump(
            data=dict(data),
            stream=f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


toml_suffixes = {".toml"}
yaml_suffixes = {".yaml", ".yml"}
supported_suffixes = toml_suffixes | yaml_suffixes


def dict_from_file(file: Path, *, encoding: str | None = "utf-8") -> dict[str, Any]:
    """Load a dict from a TOML or YAML file (dispatched by suffix).

    Args:
        file: Config file path.
        encoding: Text encoding for YAML (ignored for TOML).

    Returns:
        Loaded mapping.
    """
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        return dict_from_toml(file)
    if suffix in yaml_suffixes:
        return dict_from_yaml(file, encoding=encoding)
    raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")


def dict_to_file(data: Mapping[str, Any], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Write a mapping to a TOML or YAML file (dispatched by suffix)."""
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        dict_to_toml(data, file)
    elif suffix in yaml_suffixes:
        dict_to_yaml(data, file, encoding=encoding)
    else:
        raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")
