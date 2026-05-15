"""Dataclass serialization utilities for TOML and YAML config files.

NeuroX config files are written in TOML (default) or YAML; both preserve
comments, which is essential for reviewable hardware parameter sets.  JSON
is intentionally not supported.

Every NeuroX runtime config is a frozen dataclass, so the load/dump path
has exactly one shape: dict ↔ dataclass.  ``Enum`` fields are encoded as
their ``.value``; generic container fields (``list[T]``, ``dict[K, V]``,
``tuple[...]``) recurse element-wise using the annotated types.

The module is layered as follows:

- ``dataclass_from_dict`` / ``dataclass_to_dict``: recursively convert
  between a dataclass tree and plain dicts suitable for TOML/YAML.  Uses
  ``typing.get_type_hints`` so nested configs (e.g.
  ``RRAMConfig.prog_gamma: StateDependentGammaConfig``) are built in one
  pass.  Unknown keys are silently dropped so older config files stay
  compatible with added fields.
- ``merge_dicts``: deep-overlay multiple dicts in descending priority
  order; used to layer a user override on top of a default config.
- ``dict_from_toml`` / ``dict_from_yaml``: format-specific dict loaders.
- ``dict_from_file`` / ``dataclass_from_file`` / ``dataclass_to_file``:
  format-agnostic entry points dispatching on file suffix (``.toml``,
  ``.yaml``, ``.yml``).  ``dataclass_from_file`` accepts multiple files
  (merged in descending priority) and an optional ``section`` to pluck a
  sub-table from a multi-config file.
- ``dict_configs_from_file`` / ``dict_configs_to_file``: helpers for
  multi-config files where each top-level table maps to a distinct
  dataclass type.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import tomli_w
import yaml

T = TypeVar("T")


# --- type helpers --- #


def _is_dataclass_type(tp: Any) -> bool:  # noqa: ANN401
    return isinstance(tp, type) and is_dataclass(tp)


def _is_enum_type(tp: Any) -> bool:  # noqa: ANN401
    return isinstance(tp, type) and issubclass(tp, Enum)


def _dataclass_field_names(cls: Any) -> set[str]:  # noqa: ANN401
    """Return the declared field names of a dataclass type.

    Takes ``Any`` so callers with a ``type[T]`` unbound to
    ``DataclassInstance`` can invoke it without casts.  The caller must
    have checked ``_is_dataclass_type`` first.
    """
    return {f.name for f in fields(cls)}


# --- dict -> dataclass --- #


def _build_value(value: Any, tp: Any) -> Any:  # noqa: ANN401
    """Coerce ``value`` into the annotated type ``tp`` recursively.

    Leaves primitives untouched.  Resolves unions by trying each arm in
    order and returning the first one that accepts the value.
    """
    origin = get_origin(tp)
    args = get_args(tp)

    # --- unions (includes Optional) --- #
    if origin is Union or origin is UnionType:
        if value is None and NoneType in args:
            return None
        last_exc: Exception | None = None
        for arg in args:
            if arg is NoneType:
                continue
            try:
                return _build_value(value, arg)
            except (TypeError, ValueError) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return value

    # --- nested dataclass --- #
    if _is_dataclass_type(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping for {tp.__name__}, got {type(value).__name__}")
        return dataclass_from_dict(tp, value)

    # --- enum --- #
    if _is_enum_type(tp):
        if isinstance(value, tp):
            return value
        return tp(value)

    # --- generic containers --- #
    if origin in (list, tuple, set, frozenset):
        if not args:
            return value
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build_value(x, args[0]) for x in value)
        if origin is tuple:
            return tuple(_build_value(x, a) for x, a in zip(value, args, strict=False))
        inner = args[0]
        return origin(_build_value(x, inner) for x in value)

    if origin is dict:
        if len(args) < 2:
            return value
        v_tp = args[1]
        return {k: _build_value(v, v_tp) for k, v in value.items()}

    # --- primitive / untyped --- #
    return value


def dataclass_from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
    """Build a dataclass instance from a mapping.

    Nested dataclass and ``Enum`` fields are resolved recursively from
    the annotated types.  Unknown keys are ignored so older config files
    stay compatible with added fields.

    Args:
        cls: Target frozen dataclass type.
        data: Source mapping.

    Returns:
        Instance of ``cls``.
    """
    if not _is_dataclass_type(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass type")
    hints = get_type_hints(cls)
    names = _dataclass_field_names(cls)
    kwargs: dict[str, Any] = {}
    for name, raw in data.items():
        if name not in names:
            continue
        kwargs[name] = _build_value(raw, hints.get(name, Any))
    return cls(**kwargs)


# --- dataclass -> dict --- #


def _to_primitive(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert a dataclass tree to primitive Python values."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_primitive(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_to_primitive(x) for x in obj]
    return obj


def dataclass_to_dict(obj: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert a dataclass instance to a plain dict.

    ``Enum`` values are written as their ``.value`` so the output is TOML
    and YAML serializable without custom representers.

    Args:
        obj: Frozen dataclass instance.

    Returns:
        Nested dict ready for TOML / YAML dumping.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{type(obj).__name__} is not a dataclass instance")
    result = _to_primitive(obj)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict result, got {type(result).__name__}")
    return result


# --- dict merging --- #


def _deep_fill_defaults(override: dict[str, Any], default: dict[str, Any], strict_type: bool) -> dict[str, Any]:
    """Fill missing keys in ``override`` from ``default`` recursively."""
    merged = override.copy()
    for k, d_v in default.items():
        if k not in merged:
            merged[k] = d_v
            continue
        o_v = merged[k]
        if isinstance(o_v, dict) and isinstance(d_v, dict):
            merged[k] = _deep_fill_defaults(o_v, d_v, strict_type=strict_type)
        elif strict_type and isinstance(o_v, dict) != isinstance(d_v, dict):
            raise ValueError(
                f"Cannot merge key '{k}': type mismatch. override={type(o_v).__name__}, default={type(d_v).__name__}",
            )
    return merged


def merge_dicts(*dicts: dict[str, Any], strict_type: bool = True) -> dict[str, Any]:
    """Deep-merge dicts from left (highest priority) to right.

    Args:
        dicts: Dicts ordered by descending priority.
        strict_type: Reject dict/non-dict conflicts when ``True``.

    Returns:
        New merged dict; inputs are not modified.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0].copy()

    merged = dicts[0].copy()
    for d in dicts[1:]:
        merged = _deep_fill_defaults(merged, d, strict_type=strict_type)
    return merged


# --- format-specific loaders / dumpers --- #


def dict_from_toml(file: Path) -> dict[str, Any]:
    """Load a dict from a TOML file.

    TOML files are read in binary mode as required by ``tomllib``.
    """
    with file.open(mode="rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"TOML root must be a table, got {type(data).__name__}")
    return data


def dict_to_toml(data: Mapping[str, Any], file: Path) -> None:
    """Write a mapping to a TOML file.

    ``None`` values are stripped recursively because TOML has no null.
    """
    with file.open(mode="wb") as f:
        tomli_w.dump(_strip_none(data), f)


def _strip_none(data: Any) -> Any:  # noqa: ANN401
    """Recursively drop ``None`` values so TOML serialization never fails."""
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
    """Write a mapping to a YAML file preserving insertion order."""
    with file.open(mode="w", encoding=encoding) as f:
        yaml.dump(
            data=dict(data),
            stream=f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


# --- file interface --- #

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


def _pluck_section(data: dict[str, Any], section: str | None) -> dict[str, Any]:
    if section is None:
        return data
    if section not in data:
        raise KeyError(f"Section '{section}' not found in config file (keys: {sorted(data)})")
    sub = data[section]
    if not isinstance(sub, dict):
        raise TypeError(f"Section '{section}' must be a table, got {type(sub).__name__}")
    return sub


def dataclass_from_file(
    cls: type[T],
    *files: Path,
    section: str | None = None,
    encoding: str | None = "utf-8",
    strict_type: bool = True,
) -> T:
    """Load a dataclass from one or more config files.

    When several files are given they are merged in descending priority
    order (the first file wins on conflicts).  When ``section`` is given,
    the same sub-table is plucked from each file before merging, so a
    user override file can target a single config inside a multi-config
    default file.

    Args:
        cls: Target frozen dataclass type.
        files: Config file paths, ordered by descending priority.
        section: Optional top-level table name to extract from each file.
        encoding: YAML text encoding (ignored for TOML).
        strict_type: Reject dict/non-dict conflicts during merge.

    Returns:
        Instance of ``cls`` built from the merged data.
    """
    if not files:
        raise ValueError("At least one config file must be provided")
    raw = [_pluck_section(dict_from_file(f, encoding=encoding), section) for f in files]
    merged = merge_dicts(*raw, strict_type=strict_type)
    return dataclass_from_dict(cls, merged)


def dataclass_to_file(obj: Any, file: Path, *, encoding: str | None = "utf-8") -> None:  # noqa: ANN401
    """Write a dataclass instance to a TOML or YAML file."""
    dict_to_file(dataclass_to_dict(obj), file, encoding=encoding)


# --- multi-config helpers --- #


def dict_configs_from_file(
    specs: Mapping[str, type],
    *files: Path,
    encoding: str | None = "utf-8",
    strict_type: bool = True,
) -> dict[str, Any]:
    """Load several configs from one or more multi-config files.

    Each key of ``specs`` names a top-level table in the file; the
    corresponding value is the dataclass type to build from that table.
    Missing sections raise ``KeyError``.

    Example::

        from neurox.config import DEFAULT_1T1R_TOML

        configs = dict_configs_from_file(
            {"rram": RRAMConfig, "bl_wire": WireConfig},
            DEFAULT_1T1R_TOML,
        )
        rram_config = configs["rram"]

    Args:
        specs: Mapping of section name to target dataclass type.
        files: Config files ordered by descending priority (merged).
        encoding: YAML text encoding.
        strict_type: Reject dict/non-dict conflicts during merge.

    Returns:
        Dict mapping each section name to its built instance.
    """
    if not files:
        raise ValueError("At least one config file must be provided")
    raw = [dict_from_file(f, encoding=encoding) for f in files]
    merged = merge_dicts(*raw, strict_type=strict_type)
    out: dict[str, Any] = {}
    for section, cls in specs.items():
        if section not in merged:
            raise KeyError(f"Section '{section}' not found in config file (keys: {sorted(merged)})")
        sub = merged[section]
        if not isinstance(sub, dict):
            raise TypeError(f"Section '{section}' must be a table, got {type(sub).__name__}")
        out[section] = dataclass_from_dict(cls, sub)
    return out


def dict_configs_to_file(
    objs: Mapping[str, Any],
    file: Path,
    *,
    encoding: str | None = "utf-8",
) -> None:
    """Write several configs to one multi-config file.

    Each ``objs`` entry becomes a top-level table named by its key.

    Args:
        objs: Mapping of section name to dataclass instance.
        file: Output file path (dispatched by suffix).
        encoding: YAML text encoding.
    """
    data = {name: dataclass_to_dict(obj) for name, obj in objs.items()}
    dict_to_file(data, file, encoding=encoding)
