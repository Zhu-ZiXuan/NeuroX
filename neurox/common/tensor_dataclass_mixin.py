"""Construction and field traversal for tensor-carrying data classes.

What a tensor field costs a data class — equality, freezing, the keyword-only
call — is settled here once for classes that opt into this hierarchy.
The free walker functions rebuild arbitrary nested dataclass instances while
mapping either one tensor tree or a corresponding pair.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeGuard, cast, dataclass_transform

from torch import Tensor

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass_transform(eq_default=False, frozen_default=True, kw_only_default=True)
class TensorDataClassMixin:
    """Supply identity equality and immutable construction to tensor data.

    A tensor compares elementwise, so a field holding one leaves value `==`
    ill-defined: the comparison returns a tensor rather than a verdict. Every
    class here therefore equals only itself, and `==` and `hash()` use identity
    throughout the hierarchy.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass`, define `__init__`, or define `__post_init__`;
    this mixin supplies a frozen, keyword-only dataclass to every descendant,
    however deep. The transformation completes before delegating to later
    class-initialization hooks, so they can inspect the complete fields.
    """

    def __init_subclass__(cls) -> None:
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} carries data only; it declares fields, not __post_init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclasses.dataclass(eq=False, frozen=True, kw_only=True)(cls)
        super().__init_subclass__()


def _is_dataclass_instance(obj: object) -> TypeGuard[DataclassInstance]:
    return not isinstance(obj, type) and dataclasses.is_dataclass(obj)


def walk_single_tensor_fields[NodeT](fn: Callable[[Tensor], Tensor], node: NodeT) -> NodeT:
    """Rebuild a dataclass by transforming its tensor fields in declaration order.

    Equivalent Python control flow:

    ```python
    return fn(node)
    ```

    Args:
        fn: Receives each tensor field and returns its replacement; owns any
            tensor mutation.
        node: Dataclass instance supporting reconstruction, including any
            nested dataclasses.

    Returns:
        Rebuilt dataclass with the same concrete types. Non-tensor fields,
        including `None`, remain unchanged.

    Raises:
        TypeError: `node` is not a dataclass instance.
    """
    if not _is_dataclass_instance(node):
        raise TypeError(f"walk_single_tensor_fields() requires a dataclass instance, got {type(node).__name__}")

    replacements: dict[str, object] = {}
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = fn(value)
        elif _is_dataclass_instance(value):
            replacements[field.name] = walk_single_tensor_fields(fn, value)
    return cast(NodeT, dataclasses.replace(node, **replacements))


def walk_paired_tensor_fields[NodeT](
    fn: Callable[[Tensor, Tensor], Tensor],
    node: NodeT,
    other: NodeT,
) -> NodeT:
    """Rebuild a dataclass by transforming corresponding tensor fields of a pair.

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
        Rebuilt dataclass with the same concrete types. Non-tensor fields,
        including `None`, retain their values from `node`.

    Raises:
        TypeError: An input is not a dataclass instance or the inputs have
            incompatible concrete structures.
    """
    if not _is_dataclass_instance(node) or not _is_dataclass_instance(other) or type(other) is not type(node):
        raise TypeError(
            f"walk_paired_tensor_fields() requires two dataclass instances with the same concrete structure, "
            f"got {type(node).__name__} and {type(other).__name__}"
        )

    replacements: dict[str, object] = {}
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        other_value = getattr(other, field.name)
        if isinstance(value, Tensor):
            if not isinstance(other_value, Tensor):
                raise TypeError("walk_paired_tensor_fields() inputs must have corresponding tensor fields")
            replacements[field.name] = fn(value, other_value)
        elif _is_dataclass_instance(value):
            replacements[field.name] = walk_paired_tensor_fields(fn, value, other_value)
        elif isinstance(other_value, Tensor) or _is_dataclass_instance(other_value):
            raise TypeError("walk_paired_tensor_fields() inputs must have the same concrete dataclass structure")
    return cast(NodeT, dataclasses.replace(node, **replacements))
