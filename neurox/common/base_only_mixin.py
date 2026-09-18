"""Explicit construction restrictions for inheritance-only classes."""

from __future__ import annotations

from typing import ClassVar, Self, final


class BaseOnlyMixin:
    """Declare inheritance-only classes with `base_only=True`.

    The restriction applies only to the declaring class; subclasses opt in
    independently.
    """

    __base_only: ClassVar[bool] = True

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__base_only = base_only

    # Python passes each concrete constructor's arguments through __new__ first.
    # Only cls reaches object.__new__; Python supplies the original arguments
    # to the concrete __init__ after allocation.
    @final
    def __new__(cls, *args: object, **kwargs: object) -> Self:
        if cls.__base_only:
            raise TypeError(f"{cls.__qualname__} is declared base_only; construct a concrete subclass instead")
        return super().__new__(cls)
