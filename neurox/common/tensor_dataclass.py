"""Shared construction semantics for tensor-carrying data classes.

What a tensor field costs a data class — equality, freezing, the keyword-only
call — is settled here once for classes that opt into this hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import dataclass_transform


@dataclass_transform(eq_default=False, frozen_default=True, kw_only_default=True)
@dataclass(eq=False, frozen=True, kw_only=True)
class TensorDataClassBase:
    """Supply identity equality and immutable construction to tensor data.

    A tensor compares elementwise, so a field holding one leaves value `==`
    ill-defined: the comparison returns a tensor rather than a verdict. Every
    class here therefore equals only itself, and `==` and `hash()` use identity
    throughout the hierarchy.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass`, define `__init__`, or define `__post_init__`;
    this base supplies a frozen, keyword-only dataclass to every descendant,
    however deep.
    """

    def __init_subclass__(cls) -> None:
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} carries data only; it declares fields, not __post_init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        super().__init_subclass__()
        dataclass(eq=False, frozen=True, kw_only=True)(cls)
