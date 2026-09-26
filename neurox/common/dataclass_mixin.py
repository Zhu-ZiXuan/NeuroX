"""Construction, PyTree registration, and traversal for tensor data classes.

Equivalent Python control flow blocks are pseudocode. Tensor notation stands
for the same operation on every tensor in a structured value; tree traversal
is omitted.
"""

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
    """Supply immutable tensor data with identity equality and hashing.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__`. This mixin supplies a
    frozen, keyword-only dataclass to every descendant before delegating to
    later class-initialization hooks, so they can inspect the complete fields.

    Field immutability prevents rebinding attributes; it does not prevent
    in-place tensor writes. Do not use value equality on these containers to
    compare numerical results; compare their tensor fields explicitly.
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


# ### Tensor field traversal ###


def _is_dataclass_instance(obj: object) -> TypeGuard[DataclassInstance]:
    # Optional trace fields resolve before Dynamo's dataclass introspection.
    if obj is None:
        return False
    return not isinstance(obj, type) and dataclasses.is_dataclass(obj)


def map_single_tensor_fields[NodeT: DataclassInstance](
    fn: Callable[[Tensor], Tensor],
    node: NodeT,
) -> NodeT:
    """Apply a callback to every direct or nested dataclass tensor field.

    Traversal follows field declaration order and recurses into dataclass
    instances only. Lists, tuples, and mappings are retained without descending
    into them. Each field occurrence is visited, including repeated references
    to one tensor. The helper performs no tensor copy or device conversion by
    itself; the callback controls replacement and mutation.

    Equivalent Python control flow:

    ```python
    return fn(node)
    ```

    Args:
        fn: Transform one tensor into its replacement.
        node: Dataclass instance whose field structure is preserved.

    Returns:
        A reconstructed instance of the same concrete type when any tensor is
        replaced. Non-tensor fields and tensor-free subtrees retain their values
        and identities. Reconstruction invokes the dataclass constructor.
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
    """Apply a callback to matching tensor fields in two dataclass trees.

    Both inputs must have the same concrete dataclass types and matching tensor
    positions. Traversal follows `node` in declaration order and recurses only
    into dataclass instances; containers are not traversed. Tensor placement,
    shape compatibility, and mutation are the callback's responsibility.

    Equivalent Python control flow:

    ```python
    return fn(node, other)
    ```

    Args:
        fn: Receives `(node_tensor, other_tensor)` and returns the replacement.
        node: Tree supplying the result's structure and non-tensor field values.
        other: Tree supplying the second tensor at each matching position.

    Returns:
        A reconstructed tree of the same concrete type, preserving tensor-free
        subtrees from `node`. Dataclass constructors run during reconstruction.

    Raises:
        TypeError: Matching dataclass nodes have different concrete types.
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
    """Visit tensor fields without reconstructing or copying the dataclass tree.

    The callback receives each tensor field in declaration order, recursively
    through nested dataclass instances. Containers such as lists and mappings
    are not traversed. Repeated references are visited at each field occurrence.
    The callback owns any tensor mutation; frozen fields do not freeze their
    underlying tensor storage.

    Equivalent Python control flow:

    ```python
    fn(node)
    ```

    Args:
        fn: Inspect or act on one tensor; its return value is discarded.
        node: Dataclass instance supplying the fields to visit.
    """
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Tensor):
            fn(value)
        elif _is_dataclass_instance(value):
            visit_tensor_fields(fn, value)
