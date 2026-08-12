"""Shared per-tensor-field dataclass traversal."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from torch import Tensor

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

NodeT = TypeVar("NodeT")


def walk_tensor_fields(node: NodeT, transform: Callable[[Tensor], Tensor]) -> NodeT:
    """Rebuild a dataclass with ``transform`` applied to every tensor field.

    The traversal core every dataclass-of-tensors rebuild in the codebase
    shares: walks ``dataclasses.fields(node)`` in declaration order, passing
    each ``Tensor`` field through ``transform`` and recursing into each
    nested dataclass-valued field; every other field (including ``None``)
    carries through unchanged via ``dataclasses.replace``. A read-only walk
    reaches the same traversal by giving ``transform`` a side effect and
    discarding the rebuilt return.

    Args:
        node: Frozen dataclass instance to walk.
        transform: Per-tensor-field transform, applied independently to
            each leaf.

    Returns:
        A new instance of ``type(node)`` with every tensor field replaced.
    """
    instance = cast("DataclassInstance", node)
    replacements: dict[str, object] = {}
    for field in dataclasses.fields(instance):
        value = getattr(node, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = transform(value)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            replacements[field.name] = walk_tensor_fields(value, transform)
    return cast(NodeT, dataclasses.replace(instance, **replacements))
