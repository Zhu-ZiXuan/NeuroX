"""The `TensorGroupMixin` per-tensor-field shape-operation surface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from torch import Tensor

from .tensor_fields import walk_tensor_fields


class TensorGroupMixin:
    """Add per-tensor-field shape operations to a dataclass of tensors.

    The host must be a dataclass, and every tensor field of it is assumed to
    share one shape — an invariant these methods rely on but never check. A
    nested dataclass field recurses, and any other field (including `None`)
    carries through unchanged. `flatten_axes` / `index_select` dims follow
    `Tensor`'s own convention: negative values are customary, not validated
    here.
    """

    def map_tensors(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Rebuild with one transform applied independently to every tensor field.

        The operation every other method here composes: type-preserving via
        `dataclasses.replace`, recursing into nested dataclass fields.

        Args:
            fn: Per-tensor-field transform.

        Returns:
            A new instance of the same type with every tensor field replaced.
        """
        return walk_tensor_fields(self, fn)

    def expand(self, shape: tuple[int, ...]) -> Self:
        """Expand every tensor field to one shape, with `Tensor.expand` semantics."""
        return self.map_tensors(lambda t: t.expand(shape))

    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Flatten every tensor field's `[start_dim, end_dim]` axes into one."""
        return self.map_tensors(lambda t: t.flatten(start_dim, end_dim))

    def index_select(self, dim: int, index: Tensor) -> Self:
        """Index-select one dim of every tensor field."""
        return self.map_tensors(lambda t: t.index_select(dim, index))
