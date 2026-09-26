"""Dict <-> file I/O for TOML and YAML."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

import tomli_w
import yaml

from .value import ConfigDict, ConfigValue, normalize_config_dict


def dict_from_toml(file: Path) -> ConfigDict:
    """Parse UTF-8 TOML into a normalized configuration mapping.

    Return string-keyed mappings and lists containing only null, boolean,
    integer, float, or string leaves. TOML date/time objects are rejected by
    normalization; quote them when they represent configuration strings. No
    composition directives or dataclass fields are interpreted. File and parse
    errors propagate.

    Args:
        file: Source file to parse.

    Returns:
        A normalized string-keyed configuration mapping.
    """
    with file.open(mode="rb") as f:
        data: object = tomllib.load(f)
    return normalize_config_dict(data)


def dict_to_toml(data: Mapping[str, ConfigValue], file: Path) -> None:
    """Write a mapping to a TOML file.

    Mapping entries with `None` values are omitted recursively because TOML has
    no null literal. List entries are retained.

    The destination is replaced in UTF-8; create its parent directory first.
    This is a direct write, not an atomic replacement, so a serialization or I/O
    failure after opening can leave a partial file. The input mapping is
    unchanged.

    Args:
        data: String-keyed mapping of supported configuration values.
        file: Destination file to replace; its parent must already exist.

    Raises:
        TypeError: A list contains `None`.
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
    """Parse YAML safely into a normalized configuration mapping.

    The document must be a mapping with string keys. Only null, boolean,
    integer, float, string, list, and mapping values are accepted after parsing.
    Quote dates and other scalars that YAML would otherwise turn into
    unsupported objects. This loader does not expand composition directives or
    construct dataclasses. File, YAML syntax, and normalization errors
    propagate.

    Args:
        file: Source file to parse.
        encoding: YAML text encoding; None selects the platform default.

    Returns:
        A normalized string-keyed configuration mapping.
    """
    with file.open(mode="r", encoding=encoding) as f:
        data: object = yaml.safe_load(f)
    return normalize_config_dict(data)


def dict_to_yaml(data: Mapping[str, ConfigValue], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Replace a file with a normalized YAML configuration mapping.

    Create the parent directory first. Values must use the supported
    configuration tree types; null values are retained. Writing preserves
    mapping order and uses `encoding`. This direct writer is not atomic: an I/O
    failure can leave a partial file. It does not mutate the supplied mapping.

    Args:
        data: String-keyed mapping of supported configuration values.
        file: Destination file to replace; its parent must already exist.
        encoding: YAML text encoding; None selects the platform default.
    """
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
    """Load a dictionary from a TOML or YAML file selected by suffix.

    Suffix matching is case-insensitive. The document must normalize to a
    string-keyed mapping of supported configuration values. This call performs
    plain parsing only; use `load_config_dict` to expand references, select
    sections, and merge multiple files.

    Args:
        file: Source file to parse.
        encoding: YAML text encoding; ignored for TOML.

    Returns:
        A normalized configuration mapping without directive expansion.

    Raises:
        ValueError: The file suffix is unsupported.
    """
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        return dict_from_toml(file)
    if suffix in yaml_suffixes:
        return dict_from_yaml(file, encoding=encoding)
    raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")


def dict_to_file(data: Mapping[str, ConfigValue], file: Path, *, encoding: str | None = "utf-8") -> None:
    """Write a mapping to a TOML or YAML file selected by suffix.

    The file is overwritten directly and parent directories must already exist.
    TOML drops null-valued mapping entries and rejects null list items; YAML
    preserves nulls. Normalization and writer errors propagate, and an
    interrupted write may leave a partial destination. Use an external
    atomic-write context when replacement must preserve an earlier artifact on
    failure.

    Args:
        data: String-keyed mapping of supported configuration values.
        file: Destination file to replace; its parent must already exist.
        encoding: YAML text encoding; ignored for TOML.

    Raises:
        ValueError: The file suffix is unsupported.
    """
    suffix = file.suffix.lower()
    if suffix in toml_suffixes:
        dict_to_toml(data, file)
    elif suffix in yaml_suffixes:
        dict_to_yaml(data, file, encoding=encoding)
    else:
        raise ValueError(f"Unsupported config suffix '{file.suffix}'. Supported: {sorted(supported_suffixes)}")
