"""Shared per-tensor-field dataclass traversal and the ``TensorGroupMixin`` surface.

See also:
    docs/internals/common/tensor_group.md
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Self, TypeVar, cast

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


class TensorGroupMixin:
    """Add per-tensor-field shape operations to a dataclass of tensors.

    Every tensor field of the host is assumed to share one shape (an
    invariant these methods rely on but never check); a nested dataclass
    field recurses, and any other field (including ``None``) carries
    through unchanged. ``flatten_axes`` / ``index_select`` dims follow
    ``Tensor``'s own convention: negative values are customary, not
    validated here.

    Host requirements:
        - Be a dataclass type.
    """

    def map_tensors(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Rebuild with ``fn`` applied independently to every tensor field.

        The operation every other method here composes: type-preserving via
        ``dataclasses.replace``, recursing into nested dataclass fields.

        Args:
            fn: Per-tensor-field transform.

        Returns:
            A new instance of the same type with every tensor field
            replaced.
        """
        return walk_tensor_fields(self, fn)

    def expand(self, shape: tuple[int, ...]) -> Self:
        """Expand every tensor field to ``shape`` (``Tensor.expand`` semantics)."""
        return self.map_tensors(lambda t: t.expand(shape))

    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Flatten every tensor field's ``[start_dim, end_dim]`` axes into one."""
        return self.map_tensors(lambda t: t.flatten(start_dim, end_dim))

    def index_select(self, dim: int, index: Tensor) -> Self:
        """Index-select ``dim`` of every tensor field with ``index``."""
        return self.map_tensors(lambda t: t.index_select(dim, index))
