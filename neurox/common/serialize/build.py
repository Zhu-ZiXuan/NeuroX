"""Convert between mappings and dataclass trees."""

from __future__ import annotations

import inspect
import typing
from abc import ABC
from collections.abc import Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import ClassVar, Protocol, TypeAliasType, TypeGuard, Union, get_args, get_origin, get_type_hints

from .keys import CLASS_DISCRIMINATOR
from .value import ConfigDict, ConfigValue, normalize_config_dict

type _PrimitiveType = type[bool] | type[int] | type[float] | type[str]


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]


def _is_dataclass_type(tp: object) -> TypeGuard[type[_DataclassInstance]]:
    return isinstance(tp, type) and is_dataclass(tp)


def _is_dataclass_instance(obj: object) -> TypeGuard[_DataclassInstance]:
    return not isinstance(obj, type) and is_dataclass(obj)


def _is_enum_type(tp: object) -> TypeGuard[type[Enum]]:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _is_primitive_type(tp: object) -> TypeGuard[_PrimitiveType]:
    return tp in (bool, int, float, str)


def _dataclass_field_names(cls: type[object]) -> set[str]:
    if not _is_dataclass_type(cls):
        raise TypeError(f"{type(cls).__name__} is not a dataclass type")
    return set(cls.__dataclass_fields__)


def _required_field_names(cls: type[object]) -> set[str]:
    """Return the field names a caller must supply, i.e. those carrying no default."""
    if not _is_dataclass_type(cls):
        raise TypeError(f"{type(cls).__name__} is not a dataclass type")
    return {f.name for f in fields(cls) if f.init and f.default is MISSING and f.default_factory is MISSING}


def _recursive_dataclass_descendants(base: type[object]) -> list[str]:
    out: list[str] = []
    for sub in base.__subclasses__():
        if _is_dataclass_type(sub):
            out.append(sub.__name__)
        out.extend(_recursive_dataclass_descendants(sub))
    return out


def _resolve_concrete_dataclass[T](base: type[T], type_name: str) -> type[T]:
    """Find one named class among a base and its recursive dataclass subclasses.

    Raises:
        TypeError: No candidate has the given name.
    """
    if base.__name__ == type_name:
        return base
    for sub in base.__subclasses__():
        if sub.__name__ == type_name and is_dataclass(sub):
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


def _build_value(value: ConfigValue, tp: object, *, path: str) -> object:
    """Coerce one value recursively into an annotated type."""
    if isinstance(tp, TypeAliasType):
        return _build_value(value, tp.__value__, path=path)

    origin = get_origin(tp)
    args = get_args(tp)

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
            if not isinstance(type_name, str):
                raise TypeError(f"{path}.{CLASS_DISCRIMINATOR}: expected str, got {type(type_name).__name__}")
            concrete = _resolve_concrete_dataclass(tp, type_name)
            filtered = {k: v for k, v in value.items() if k != CLASS_DISCRIMINATOR}
            return _dataclass_from_config_dict(concrete, filtered)
        return _dataclass_from_config_dict(tp, value)

    # Enum.
    if _is_enum_type(tp):
        return tp(value)

    # Generic containers.
    if origin in (list, tuple, set, frozenset):
        if not isinstance(value, list):
            raise TypeError(f"Expected list for {tp}, got {type(value).__name__}: {value!r}")
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
        items = [_build_value(x, inner, path=f"{path}[{index}]") for index, x in enumerate(value)]
        if origin is list:
            return items
        if origin is set:
            return set(items)
        return frozenset(items)

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
    if _is_primitive_type(tp):
        return _coerce_primitive(value, tp)

    if tp is Path:
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected path string, got {type(value).__name__}")
        return Path(value)

    raise TypeError(f"{path}: unsupported field annotation {tp!r}")


def _coerce_primitive(value: ConfigValue, tp: _PrimitiveType) -> bool | int | float | str:
    """Validate a primitive value against its declared type, without silent coercion."""
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
    raise TypeError(f"unsupported primitive type {tp!r}")


def dataclass_from_dict[T](cls: type[T], data: Mapping[str, ConfigValue]) -> T:
    """Build a dataclass instance from a mapping.

    Nested dataclass and `Enum` fields are resolved recursively. A top-level
    `_neurox_class` discriminator dispatches to the named subclass of `cls`; it
    resolves only within `cls` and its subclasses, so the receiver bounds what
    the data can construct. The abstract-base rejection applies wherever such a
    base appears: as `cls` itself, as the class a discriminator names, or as a
    base-typed nested field.

    Returns:
        Instance of `cls`, or of its named subclass.

    Raises:
        TypeError: `cls` is not a dataclass, `data` names a `_neurox_class` that
            is neither `cls` nor a subclass of it, `data` carries a key that
            matches no field of the resolved class, `data` omits a field the
            resolved class declares without a default, a value does not match its
            field's declared primitive type (only an `int` widens to a `float`),
            or the resolved class is an abstract config base (declares `ABC` as
            a direct base or has unimplemented abstract methods) rather than a
            concrete class. A concrete class stays buildable even when
            subclasses of it exist elsewhere.
    """
    return _dataclass_from_config_dict(cls, normalize_config_dict(data))


def _dataclass_from_config_dict[T](cls: type[T], data: ConfigDict) -> T:
    """Build a dataclass from an already validated configuration mapping."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass type")
    type_name = data.get(CLASS_DISCRIMINATOR)
    if type_name is not None:
        if not isinstance(type_name, str):
            raise TypeError(f"{CLASS_DISCRIMINATOR}: expected str, got {type(type_name).__name__}")
        concrete = _resolve_concrete_dataclass(cls, type_name)
        if concrete is not cls:
            filtered = {k: v for k, v in data.items() if k != CLASS_DISCRIMINATOR}
            return _dataclass_from_config_dict(concrete, filtered)
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
    missing = _required_field_names(cls) - set(data)
    if missing:
        raise TypeError(f"{cls.__name__}: missing key(s) {sorted(missing)}; valid fields: {sorted(names)}")
    kwargs: dict[str, object] = {}
    for name, raw in data.items():
        if name == CLASS_DISCRIMINATOR:
            continue
        if name not in hints:
            raise TypeError(f"{cls.__name__}.{name}: field annotation is unavailable")
        kwargs[name] = _build_value(raw, hints[name], path=f"{cls.__name__}.{name}")
    return cls(**kwargs)


def _is_polymorphic_dataclass(tp: type[_DataclassInstance]) -> bool:
    if any(_is_dataclass_type(base) and base is not tp for base in tp.__mro__):
        return True
    return any(_is_dataclass_type(sub) for sub in tp.__subclasses__())


def _to_primitive(obj: object) -> ConfigValue:
    """Recursively convert a dataclass tree to primitive Python values."""
    if isinstance(obj, Enum):
        return _to_primitive(obj.value)
    if _is_dataclass_instance(obj):
        out: ConfigDict = {}
        cls = type(obj)
        if _is_polymorphic_dataclass(cls):
            out[CLASS_DISCRIMINATOR] = cls.__name__
        for name in obj.__dataclass_fields__:
            out[name] = _to_primitive(getattr(obj, name))
        return out
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"configuration mapping key must be str, got {type(key).__name__}")
            out[key] = _to_primitive(value)
        return out
    if isinstance(obj, list | tuple | set | frozenset):
        return [_to_primitive(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    raise TypeError(f"unsupported configuration value {type(obj).__name__}")


def dataclass_to_dict(obj: object) -> ConfigDict:
    """Convert a dataclass instance to a plain dict.

    Returns:
        Nested dict ready for TOML / YAML dumping.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError(f"{type(obj).__name__} is not a dataclass instance")
    result = _to_primitive(obj)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict result, got {type(result).__name__}")
    return result
