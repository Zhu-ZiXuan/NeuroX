"""Dataclass serialization utilities for TOML and YAML config files.

See also:
    docs/dev/modules/common/load_dump.md
"""

import tomllib
import typing
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import tomli_w
import yaml

T = TypeVar("T")


# --- type helpers ---


def _is_dataclass_type(tp: Any) -> bool:  # noqa: ANN401
    return isinstance(tp, type) and is_dataclass(tp)


def _is_enum_type(tp: Any) -> bool:  # noqa: ANN401
    return isinstance(tp, type) and issubclass(tp, Enum)


def _dataclass_field_names(cls: Any) -> set[str]:  # noqa: ANN401
    """Return the declared field names of a dataclass type."""
    return {f.name for f in fields(cls)}


# --- dict -> dataclass ---


_TYPE_DISCRIMINATOR = "_neurox_type"
_USE_DIRECTIVE = "_neurox_use"
_USE_PRESET_DIRECTIVE = "_neurox_use_preset"


def _resolve_concrete_dataclass(base: type, type_name: str) -> type:
    """Find a dataclass subclass of ``base`` named ``type_name``.

    ``base`` itself is considered first, then ``base.__subclasses__()`` recursively.

    Raises:
        TypeError: When no dataclass subclass of ``base`` has the given name.
    """
    if base.__name__ == type_name and _is_dataclass_type(base):
        return base
    for sub in base.__subclasses__():
        if sub.__name__ == type_name and _is_dataclass_type(sub):
            return sub
        try:
            return _resolve_concrete_dataclass(sub, type_name)
        except TypeError:
            continue
    raise TypeError(
        f"No dataclass subclass of {base.__name__} named {type_name!r}; "
        f"available subclasses (recursive): "
        f"{sorted(_recursive_dataclass_descendants(base)) or '<none>'}"
    )


def _recursive_dataclass_descendants(base: type) -> list[str]:
    """List every dataclass descendant of ``base`` by ``__name__``."""
    out: list[str] = []
    for sub in base.__subclasses__():
        if _is_dataclass_type(sub):
            out.append(sub.__name__)
        out.extend(_recursive_dataclass_descendants(sub))
    return out


def _build_value(value: Any, tp: Any) -> Any:  # noqa: ANN401
    """Coerce ``value`` into the annotated type ``tp`` recursively.

    Unions are resolved by trying each arm and returning the first that
    accepts the value. Polymorphic dataclass fields with a ``_neurox_type``
    discriminator in the value mapping instantiate the named subclass.
    ``Literal[...]`` annotations enforce membership in the declared set.
    """
    origin = get_origin(tp)
    args = get_args(tp)

    # --- Literal: enforce membership ---
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"value {value!r} not in Literal{list(args)}")
        return value

    # --- unions (includes Optional) ---
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

    # --- nested dataclass ---
    if _is_dataclass_type(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping for {tp.__name__}, got {type(value).__name__}")
        type_name = value.get(_TYPE_DISCRIMINATOR)
        if type_name is not None:
            concrete = _resolve_concrete_dataclass(tp, type_name)
            filtered = {k: v for k, v in value.items() if k != _TYPE_DISCRIMINATOR}
            return dataclass_from_dict(concrete, filtered)
        return dataclass_from_dict(tp, value)

    # --- enum ---
    if _is_enum_type(tp):
        if isinstance(value, tp):
            return value
        return tp(value)

    # --- generic containers ---
    if origin in (list, tuple, set, frozenset):
        # ``str`` / ``bytes`` are iterables of length-1 elements; if a
        # config field is annotated ``list[T]`` and the TOML supplies a
        # string by mistake, the silent character-iteration that
        # results is almost always wrong. Reject it explicitly.
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"Expected list/tuple/set for {tp}, got {type(value).__name__}: {value!r}"
            )
        if not args:
            return value
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build_value(x, args[0]) for x in value)
        if origin is tuple:
            if len(value) != len(args):
                raise ValueError(
                    f"tuple length mismatch: expected {len(args)} element(s) for "
                    f"tuple[{', '.join(getattr(a, '__name__', str(a)) for a in args)}], "
                    f"got {len(value)}"
                )
            return tuple(_build_value(x, a) for x, a in zip(value, args, strict=True))
        inner = args[0]
        return origin(_build_value(x, inner) for x in value)

    if origin is dict:
        if len(args) < 2:
            return value
        v_tp = args[1]
        return {k: _build_value(v, v_tp) for k, v in value.items()}

    # --- primitive (int / float / bool / str) ---
    if tp in (int, float, bool, str):
        return _coerce_primitive(value, tp)

    # --- untyped / Any ---
    return value


def _coerce_primitive(value: Any, tp: type) -> Any:  # noqa: ANN401
    """Validate a primitive value against ``tp``; reject silent mis-coercion.

    Rules:
      * ``bool`` is **not** a valid ``int`` here — Python's
        ``isinstance(True, int) is True`` would otherwise let bool
        fields collapse into int fields.
      * ``int`` is accepted as ``float`` (TOML / YAML round-trip
        regularly emits ``1`` where ``1.0`` is intended).
      * ``str``, ``bytes``, and any other non-numeric type are rejected
        for numeric fields with a clear ``TypeError``.
    """
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Expected bool, got {type(value).__name__}: {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}: {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool):
            raise TypeError(f"Expected float, got {type(value).__name__}: {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"Expected float, got {type(value).__name__}: {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise TypeError(f"Expected str, got {type(value).__name__}: {value!r}")
    return value


def dataclass_from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
    """Build a dataclass instance from a mapping.

    Nested dataclass and ``Enum`` fields are resolved recursively.
    A top-level ``_neurox_type`` discriminator dispatches to the named
    subclass of ``cls``. Unknown keys raise ``TypeError`` — typos must
    not silently fall back to defaults.

    Args:
        cls: Target frozen dataclass type.
        data: Source mapping.

    Returns:
        Instance of ``cls`` (or its named subclass).
    """
    if not _is_dataclass_type(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass type")
    type_name = data.get(_TYPE_DISCRIMINATOR)
    if type_name is not None:
        concrete = _resolve_concrete_dataclass(cls, type_name)
        if concrete is not cls:
            filtered = {k: v for k, v in data.items() if k != _TYPE_DISCRIMINATOR}
            return dataclass_from_dict(concrete, filtered)  # type: ignore[return-value]
    hints = get_type_hints(cls)
    names = _dataclass_field_names(cls)
    unknown = [k for k in data if k != _TYPE_DISCRIMINATOR and k not in names]
    if unknown:
        raise TypeError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}; "
            f"valid fields: {sorted(names)}"
        )
    kwargs: dict[str, Any] = {}
    for name, raw in data.items():
        if name == _TYPE_DISCRIMINATOR:
            continue
        kwargs[name] = _build_value(raw, hints.get(name, Any))
    return cls(**kwargs)


# --- dataclass -> dict ---


def _is_polymorphic_dataclass(tp: type) -> bool:
    """``True`` iff ``tp`` participates in a polymorphic family.

    A class participates whenever it has a dataclass ancestor or
    descendant — i.e., a sibling concrete type could occupy the same
    declared field. Non-polymorphic standalones (no parent dataclass,
    no subclasses) skip the ``_neurox_type`` discriminator on dump so
    the round-trip output stays terse.
    """
    if any(_is_dataclass_type(base) and base is not tp for base in tp.__mro__):
        return True
    return any(_is_dataclass_type(sub) for sub in tp.__subclasses__())


def _to_primitive(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert a dataclass tree to primitive Python values."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        cls = type(obj)
        if _is_polymorphic_dataclass(cls):
            out[_TYPE_DISCRIMINATOR] = cls.__name__
        for f in fields(obj):
            out[f.name] = _to_primitive(getattr(obj, f.name))
        return out
    if isinstance(obj, Mapping):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_to_primitive(x) for x in obj]
    return obj


def dataclass_to_dict(obj: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert a dataclass instance to a plain dict.

    ``Enum`` values are written as their ``.value``.

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


# --- dict merging ---


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


# --- format-specific loaders / dumpers ---


def dict_from_toml(file: Path) -> dict[str, Any]:
    """Load a dict from a TOML file."""
    with file.open(mode="rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"TOML root must be a table, got {type(data).__name__}")
    return data


def dict_to_toml(data: Mapping[str, Any], file: Path) -> None:
    """Write a mapping to a TOML file. ``None`` values are stripped recursively."""
    with file.open(mode="wb") as f:
        tomli_w.dump(_strip_none(data), f)


def _strip_none(data: Any) -> Any:  # noqa: ANN401
    """Recursively drop ``None`` values."""
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


# --- file interface ---

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


# --- _neurox_use cross-file references ---


def _resolve_fragment_path(rel: str, base_dir: Path) -> Path:
    """Resolve a ``_neurox_use`` relative path to an existing file.

    Suffix-free paths try ``.toml`` then ``.yaml`` / ``.yml``.
    """
    candidate = base_dir / rel
    if candidate.exists():
        return candidate
    if candidate.suffix == "":
        for suffix in (".toml", ".yaml", ".yml"):
            with_suffix = candidate.with_suffix(suffix)
            if with_suffix.exists():
                return with_suffix
    raise FileNotFoundError(f"{_USE_DIRECTIVE} fragment {rel!r} not found relative to {base_dir}")


def _parse_use_ref(ref: Any, base_dir: Path) -> tuple[Path, str]:  # noqa: ANN401
    """Parse ``"<rel_path>:<section>"`` into ``(absolute_path, section_name)``."""
    if not isinstance(ref, str):
        raise TypeError(f"{_USE_DIRECTIVE} must be a string, got {type(ref).__name__}")
    if ":" not in ref:
        raise ValueError(f"{_USE_DIRECTIVE} reference {ref!r} missing ':' (expected '<path>:<section>')")
    rel, section = ref.split(":", 1)
    if not rel or not section:
        raise ValueError(f"{_USE_DIRECTIVE} reference {ref!r} has empty path or section")
    return _resolve_fragment_path(rel, base_dir), section


def _presets_root() -> Path:
    """Return the absolute path to the ``neurox/presets/`` directory.

    Uses ``importlib.resources`` so editable installs and wheel installs both
    work; the directory's location follows wherever the ``neurox`` package is.
    """
    import importlib.resources

    return Path(str(importlib.resources.files("neurox") / "presets"))


def _validate_preset_ref_path(rel: str) -> None:
    """Reject preset paths that are absolute or try to escape the presets root."""
    if not rel:
        raise ValueError(f"{_USE_PRESET_DIRECTIVE} path is empty")
    if rel.startswith("./") or rel.startswith("/") or rel.startswith("\\"):
        raise ValueError(f"{_USE_PRESET_DIRECTIVE} path must not start with './' or be absolute: {rel!r}")
    if any(part == ".." for part in rel.replace("\\", "/").split("/")):
        raise ValueError(f"{_USE_PRESET_DIRECTIVE} path must not contain '..' segments: {rel!r}")


def _resolve_preset_fragment_path(rel: str) -> Path:
    """Resolve a preset-relative path to an existing file under ``neurox/presets/``."""
    root = _presets_root()
    candidate = root / rel
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        for suffix in (".toml", ".yaml", ".yml"):
            with_suffix = candidate.with_suffix(suffix)
            if with_suffix.is_file():
                return with_suffix
    raise FileNotFoundError(f"{_USE_PRESET_DIRECTIVE} fragment {rel!r} not found under {root}")


def _parse_preset_ref(ref: Any) -> tuple[Path, str]:  # noqa: ANN401
    """Parse a preset ``"<rel_path>:<section>"`` anchored at ``neurox/presets/``."""
    if not isinstance(ref, str):
        raise TypeError(f"{_USE_PRESET_DIRECTIVE} must be a string, got {type(ref).__name__}")
    if ":" not in ref:
        raise ValueError(f"{_USE_PRESET_DIRECTIVE} reference {ref!r} missing ':' (expected '<path>:<section>')")
    rel, section = ref.split(":", 1)
    if not rel or not section:
        raise ValueError(f"{_USE_PRESET_DIRECTIVE} reference {ref!r} has empty path or section")
    _validate_preset_ref_path(rel)
    return _resolve_preset_fragment_path(rel), section


def _resolve_directive_branch(
    value: Mapping[str, Any],
    *,
    directive: str,
    path: Path,
    section: str,
    base_dir_for_fragment: Path,
    base_dir_for_inline: Path,
    in_preset_for_fragment: bool,
    in_preset_for_inline: bool,
    cache: dict[Path, dict[str, Any]],
    in_progress: frozenset[tuple[Path, str]],
) -> Any:  # noqa: ANN401
    """Resolve one ``(directive, path, section)`` fragment-merge step.

    Shared core of the ``_neurox_use`` and ``_neurox_use_preset`` branches:
    detect cycles, load the target section, recurse into the fragment and
    the inline override under their respective ``(base_dir, in_preset)``
    contexts, then merge with inline taking priority.
    """
    key = (path, section)
    if key in in_progress:
        trail = " -> ".join(f"{p.name}:{s}" for p, s in in_progress)
        raise ValueError(f"{directive} cycle detected: {trail} -> {path.name}:{section}")
    if path not in cache:
        cache[path] = dict_from_file(path)
    root = cache[path]
    if section not in root:
        raise KeyError(f"{directive} target section {section!r} not found in {path} (keys: {sorted(root)})")
    target = root[section]
    if not isinstance(target, Mapping):
        raise TypeError(f"{directive} target {value[directive]!r} must be a table, got {type(target).__name__}")
    resolved_fragment = _resolve_uses_in_value(
        dict(target),
        base_dir_for_fragment,
        cache=cache,
        in_progress=in_progress | {key},
        in_preset=in_preset_for_fragment,
    )
    inline = {k: v for k, v in value.items() if k != directive}
    resolved_inline = _resolve_uses_in_value(
        inline,
        base_dir_for_inline,
        cache=cache,
        in_progress=in_progress,
        in_preset=in_preset_for_inline,
    )
    return merge_dicts(resolved_inline, resolved_fragment, strict_type=True)


def _resolve_uses_in_value(
    value: Any,  # noqa: ANN401
    base_dir: Path,
    *,
    cache: dict[Path, dict[str, Any]],
    in_progress: frozenset[tuple[Path, str]],
    in_preset: bool = False,
) -> Any:  # noqa: ANN401
    """Recursively resolve ``_neurox_use`` and ``_neurox_use_preset`` in ``value``.

    A mapping carrying either directive is replaced by
    ``merge_dicts(inline, fragment)``; the inline override takes priority.

    The two directives differ only in path resolution:

    - ``_neurox_use`` resolves relative to ``base_dir`` (the directory of the
      file containing the directive). Use for user-side sibling fragments.
    - ``_neurox_use_preset`` resolves relative to ``neurox/presets/``. Once
      entered, the subtree is in *preset mode* (``in_preset=True``) which
      forbids ``_neurox_use``, so preset dependency graphs stay closed inside
      the package.

    The two directives are mutually exclusive in the same sub-table. Cycles
    raise ``ValueError``.
    """
    if isinstance(value, Mapping):
        has_use = _USE_DIRECTIVE in value
        has_preset = _USE_PRESET_DIRECTIVE in value
        if has_use and has_preset:
            raise ValueError(
                f"{_USE_DIRECTIVE!r} and {_USE_PRESET_DIRECTIVE!r} are mutually exclusive in the same table"
            )
        if in_preset and has_use:
            raise ValueError(
                f"{_USE_DIRECTIVE!r} is forbidden inside neurox/presets/; use {_USE_PRESET_DIRECTIVE!r} instead"
            )
        if has_preset:
            path, section = _parse_preset_ref(value[_USE_PRESET_DIRECTIVE])
            return _resolve_directive_branch(
                value,
                directive=_USE_PRESET_DIRECTIVE,
                path=path,
                section=section,
                base_dir_for_fragment=_presets_root(),
                base_dir_for_inline=base_dir,
                in_preset_for_fragment=True,
                in_preset_for_inline=in_preset,
                cache=cache,
                in_progress=in_progress,
            )
        if has_use:
            path, section = _parse_use_ref(value[_USE_DIRECTIVE], base_dir)
            return _resolve_directive_branch(
                value,
                directive=_USE_DIRECTIVE,
                path=path,
                section=section,
                base_dir_for_fragment=path.parent,
                base_dir_for_inline=base_dir,
                in_preset_for_fragment=in_preset,
                in_preset_for_inline=in_preset,
                cache=cache,
                in_progress=in_progress,
            )
        return {
            k: _resolve_uses_in_value(v, base_dir, cache=cache, in_progress=in_progress, in_preset=in_preset)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_uses_in_value(x, base_dir, cache=cache, in_progress=in_progress, in_preset=in_preset)
            for x in value
        ]
    return value


def resolve_uses(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Expand every ``_neurox_use`` / ``_neurox_use_preset`` directive in ``data``.

    ``_neurox_use = "<rel_path>:<section>"`` resolves the path relative to
    ``base_dir`` (the directory of the file containing the directive) and
    pulls the named section from that file; inline keys override the
    fragment. ``_neurox_use_preset`` follows the same merge semantics but
    resolves paths from ``neurox/presets/`` and forbids ``_neurox_use``
    inside the preset subtree.

    Args:
        data: Loaded dict from a config file (TOML or YAML).
        base_dir: Directory for resolving relative ``_neurox_use`` paths.

    Returns:
        New dict with every directive expanded.
    """
    result = _resolve_uses_in_value(data, base_dir, cache={}, in_progress=frozenset())
    if not isinstance(result, dict):
        raise TypeError(f"directive resolution expected dict root, got {type(result).__name__}")
    return result


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

    Multiple files are merged in descending priority (first wins). When
    ``section`` is given, the same sub-table is plucked from each file
    before merging.

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
    raw = [
        _pluck_section(
            resolve_uses(dict_from_file(f, encoding=encoding), base_dir=f.parent),
            section,
        )
        for f in files
    ]
    merged = merge_dicts(*raw, strict_type=strict_type)
    return dataclass_from_dict(cls, merged)


def dataclass_to_file(obj: Any, file: Path, *, encoding: str | None = "utf-8") -> None:  # noqa: ANN401
    """Write a dataclass instance to a TOML or YAML file."""
    dict_to_file(dataclass_to_dict(obj), file, encoding=encoding)


# --- multi-config helpers ---


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

        configs = dict_configs_from_file(
            {"foo": FooConfig, "bar": BarConfig},
            Path("default.toml"),
        )
        foo_config = configs["foo"]

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
    raw = [resolve_uses(dict_from_file(f, encoding=encoding), base_dir=f.parent) for f in files]
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
