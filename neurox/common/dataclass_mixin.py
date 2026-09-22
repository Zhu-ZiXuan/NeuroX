"""Construction, PyTree registration, and traversal for tensor data classes."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeGuard, dataclass_transform

import torch
from torch import Tensor

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass_transform(eq_default=False, frozen_default=True, kw_only_default=True)
class TensorDataClassMixin:
    """Supply identity equality and immutable construction to tensor data.

    Equality and hashing use identity throughout the hierarchy.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__`. This mixin supplies a
    frozen, keyword-only dataclass to every descendant before delegating to later
    class-initialization hooks, so they can inspect the complete fields.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclasses.dataclass(eq=False, frozen=True, kw_only=True)(cls)
        super().__init_subclass__(**kwargs)


class PyTreeDataClassMixin:
    """Register dataclass hosts as PyTrees during cooperative class initialization.

    The host's dataclass transformation must complete before registration.
    Class-initialization hooks must delegate through `super`; a class decorator
    runs after these hooks and cannot supply the required transformation.

    Registration happens at class definition, before instances enter traced
    code. PyTorch exposes the constructor fields as children and records which
    optional fields are None in the tree structure. Nested dataclass types must
    also be registered to expose their fields to PyTree traversal.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        torch.export.register_dataclass(cls)


# ### walking through dataclass tensor fields ###


def _is_dataclass_instance(obj: object) -> TypeGuard[DataclassInstance]:
    return not isinstance(obj, type) and dataclasses.is_dataclass(obj)


def map_single_tensor_fields[NodeT: DataclassInstance](
    fn: Callable[[Tensor], Tensor],
    node: NodeT,
) -> NodeT:
    """Transform tensor fields of a dataclass in declaration order.

    Equivalent Python control flow:

    ```python
    return fn(node)
    ```

    Args:
        fn: Returns a replacement for each tensor field; owns any tensor mutation.
        node: Dataclass instance, including any nested dataclasses.

    Returns:
        Dataclass of the same concrete type. Non-tensor fields are preserved;
        subtrees without tensor fields retain their original objects.
    """
    replacements: dict[str, object] = {}
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = fn(value)
        elif _is_dataclass_instance(value):
            nested = map_single_tensor_fields(fn, value)
            if nested is not value:
                replacements[field.name] = nested
    return dataclasses.replace(node, **replacements) if replacements else node


def map_paired_tensor_fields[NodeT: DataclassInstance](
    fn: Callable[[Tensor, Tensor], Tensor],
    node: NodeT,
    other: NodeT,
) -> NodeT:
    """Transform corresponding tensor fields of two dataclasses.

    Equivalent Python control flow:

    ```python
    return fn(node, other)
    ```

    Args:
        fn: Receives corresponding tensors in `(node, other)` order and
            returns their replacement; owns any tensor mutation.
        node: Dataclass instance whose structure is retained.
        other: Dataclass with the same concrete nested structure and tensor
            field positions as `node`.

    Returns:
        Dataclass of the same concrete type. Non-tensor fields retain their
        values from `node`; subtrees without tensor fields retain their
        original objects from `node`.

    Raises:
        TypeError: Inputs have incompatible concrete structures.
    """
    if type(other) is not type(node):
        raise TypeError("map_paired_tensor_fields() inputs must have the same concrete dataclass structure")

    replacements: dict[str, object] = {}
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        other_value = getattr(other, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = fn(value, other_value)
        elif _is_dataclass_instance(value):
            nested = map_paired_tensor_fields(fn, value, other_value)
            if nested is not value:
                replacements[field.name] = nested
    return dataclasses.replace(node, **replacements) if replacements else node


def visit_tensor_fields(fn: Callable[[Tensor], None], node: DataclassInstance) -> None:
    """Visit tensor fields in declaration order without reconstructing objects.

    Equivalent Python control flow:

    ```python
    fn(node)
    ```

    Args:
        fn: Inspects each tensor field; owns any tensor mutation.
        node: Dataclass instance, including any nested dataclasses. Subtrees
            without tensor fields do not invoke the callback.
    """
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Tensor):
            fn(value)
        elif _is_dataclass_instance(value):
            visit_tensor_fields(fn, value)
