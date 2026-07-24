"""Convert between mappings and dataclass trees."""

from __future__ import annotations

import inspect
import typing
from abc import ABC
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from neurox.common.serialize.keys import CLASS_DISCRIMINATOR

T = TypeVar("T")


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and is_dataclass(tp)


def _is_enum_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _dataclass_field_names(cls: Any) -> set[str]:
    """Return the declared field names of a dataclass type."""
    return {f.name for f in fields(cls)}


def _recursive_dataclass_descendants(base: type) -> list[str]:
    """List every dataclass descendant of ``base`` by ``__name__``."""
    out: list[str] = []
    for sub in base.__subclasses__():
        if _is_dataclass_type(sub):
            out.append(sub.__name__)
        out.extend(_recursive_dataclass_descendants(sub))
    return out


def _resolve_concrete_dataclass(base: type, type_name: str) -> type:
    """Find a dataclass in ``{base} ∪ recursive-subclasses`` named ``type_name``.

    Raises:
        TypeError: When no dataclass in ``{base} ∪ recursive-subclasses`` has the
            given name.
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
        f"_neurox_class {type_name!r} is not {base.__name__} nor a subclass of it; "
        f"candidates in this slot (recursive): "
        f"{sorted(_recursive_dataclass_descendants(base)) or '<none>'}"
    )


def _build_value(value: Any, tp: Any, *, path: str) -> Any:
    """Coerce ``value`` recursively into the annotated type ``tp``."""
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return value

    # Literal.
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"value {value!r} not in Literal{list(args)}")
        return value

    # Union, including Optional.
    if origin is Union or origin is UnionType:
        if value is None and NoneType in args:
            return None
        last_exc: Exception | None = None
        for arg in args:
            if arg is NoneType:
                continue
            try:
                return _build_value(value, arg, path=path)
            except (TypeError, ValueError) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return value

    # Nested dataclass.
    if _is_dataclass_type(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping for {tp.__name__}, got {type(value).__name__}")
        type_name = value.get(CLASS_DISCRIMINATOR)
        if type_name is not None:
            concrete = _resolve_concrete_dataclass(tp, type_name)
            filtered = {k: v for k, v in value.items() if k != CLASS_DISCRIMINATOR}
            return dataclass_from_dict(concrete, filtered)
        return dataclass_from_dict(tp, value)

    # Enum.
    if _is_enum_type(tp):
        if isinstance(value, tp):
            return value
        return tp(value)

    # Generic containers.
    if origin in (list, tuple, set, frozenset):
        if isinstance(value, (str, bytes)):
            raise TypeError(f"Expected list/tuple/set for {tp}, got {type(value).__name__}: {value!r}")
        if not args:
            return value
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build_value(x, args[0], path=f"{path}[{index}]") for index, x in enumerate(value))
        if origin is tuple:
            if len(value) != len(args):
                raise ValueError(
                    f"tuple length mismatch: expected {len(args)} element(s) for "
                    f"tuple[{', '.join(getattr(a, '__name__', str(a)) for a in args)}], "
                    f"got {len(value)}"
                )
            return tuple(
                _build_value(x, item_type, path=f"{path}[{index}]")
                for index, (x, item_type) in enumerate(zip(value, args, strict=True))
            )
        inner = args[0]
        return origin(_build_value(x, inner, path=f"{path}[{index}]") for index, x in enumerate(value))

    if origin is dict:
        if len(args) < 2:
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}: expected mapping for {tp}, got {type(value).__name__}")
        key_type, value_type = args
        return {
            _build_value(k, key_type, path=f"{path}.<key>"): _build_value(v, value_type, path=f"{path}[{k!r}]")
            for k, v in value.items()
        }

    # Primitive.
    if tp in (int, float, bool, str):
        return _coerce_primitive(value, tp)

    if tp is Path:
        if not isinstance(value, str | Path):
            raise TypeError(f"{path}: expected path string, got {type(value).__name__}")
        return Path(value)

    raise TypeError(f"{path}: unsupported field annotation {tp!r}")


def _coerce_primitive(value: Any, tp: type) -> Any:
    """Validate a primitive value against ``tp`` without silent coercion."""
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
    A top-level ``_neurox_class`` discriminator dispatches to the named
    subclass of ``cls``; it resolves only within ``cls`` and its subclasses, so
    the receiver bounds what the data can construct. The abstract-base
    rejection applies wherever such a base appears: as ``cls`` itself, as the
    class a discriminator names, or as a base-typed nested field.

    Args:
        cls: Target frozen dataclass type.
        data: Source mapping.

    Returns:
        Instance of ``cls`` (or its named subclass).

    Raises:
        TypeError: ``cls`` is not a dataclass, ``data`` names a
            ``_neurox_class`` that is neither ``cls`` nor a subclass of it,
            ``data`` carries a key that matches no field of the resolved
            class, a value does not match its field's declared primitive type
            (only an ``int`` widens to a ``float``), or the resolved class is
            an abstract config base (declares ``ABC`` as a direct base or has
            unimplemented abstract methods) rather than a concrete class. A
            concrete class stays buildable even when subclasses of it exist
            elsewhere — abstractness is the class's own declared signal, never
            a side effect of what other packages import.
    """
    if not _is_dataclass_type(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass type")
    type_name = data.get(CLASS_DISCRIMINATOR)
    if type_name is not None:
        concrete = _resolve_concrete_dataclass(cls, type_name)
        if concrete is not cls:
            filtered = {k: v for k, v in data.items() if k != CLASS_DISCRIMINATOR}
            return dataclass_from_dict(concrete, filtered)
    if ABC in cls.__bases__ or inspect.isabstract(cls):
        descendants = sorted(_recursive_dataclass_descendants(cls))
        raise TypeError(
            f"{cls.__name__} is an abstract config base; select a concrete subclass "
            f"via the _neurox_class discriminator (one of: {descendants or '<none>'})"
        )
    hints = get_type_hints(cls)
    names = _dataclass_field_names(cls)
    unknown = [k for k in data if k != CLASS_DISCRIMINATOR and k not in names]
    if unknown:
        raise TypeError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}; valid fields: {sorted(names)}")
    kwargs: dict[str, Any] = {}
    for name, raw in data.items():
        if name == CLASS_DISCRIMINATOR:
            continue
        kwargs[name] = _build_value(raw, hints.get(name, Any), path=f"{cls.__name__}.{name}")
    return cls(**kwargs)


def _is_polymorphic_dataclass(tp: type) -> bool:
    """``True`` iff ``tp`` participates in a polymorphic family."""
    if any(_is_dataclass_type(base) and base is not tp for base in tp.__mro__):
        return True
    return any(_is_dataclass_type(sub) for sub in tp.__subclasses__())


def _to_primitive(obj: Any) -> Any:
    """Recursively convert a dataclass tree to primitive Python values."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        cls = type(obj)
        if _is_polymorphic_dataclass(cls):
            out[CLASS_DISCRIMINATOR] = cls.__name__
        for f in fields(obj):
            out[f.name] = _to_primitive(getattr(obj, f.name))
        return out
    if isinstance(obj, Mapping):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_to_primitive(x) for x in obj]
    return obj


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a plain dict.

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
