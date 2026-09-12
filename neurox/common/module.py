"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Self, dataclass_transform, final

import torch.nn as nn
from torch import Tensor

from .profile_mixin import ProfileMixin
from .pytree_dataclass_mixin import PyTreeDataClassMixin
from .serialize_mixin import SerializeMixin
from .tensor_dataclass_mixin import TensorDataClassMixin, walk_single_tensor_fields
from .validate_mixin import ValidateMixin


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class ConfigBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module configurations.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this configuration.

        Overrides call `super().validate()` before checking their own fields
        and raise `ValueError` when a constraint is violated.
        """


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class PolicyBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module runtime policies.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this runtime policy.

        Overrides call `super().validate()` before checking their own fields
        and raise `ValueError` when a constraint is violated.
        """


class SnapBase(TensorDataClassMixin, PyTreeDataClassMixin):
    """Registered snapshot PyTree with tensor transforms under one per-call layout."""

    @final
    def expand(self, shape: tuple[int, ...]) -> Self:
        """Expand every tensor field with `Tensor.expand` semantics."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.expand(shape)

        return walk_single_tensor_fields(fn, self)

    @final
    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Flatten every tensor field's `[start_dim, end_dim]` axes into one."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.flatten(start_dim, end_dim)

        return walk_single_tensor_fields(fn, self)

    @final
    def index_select(self, dim: int, index: Tensor) -> Self:
        """Index-select one dim of every tensor field."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.index_select(dim, index)

        return walk_single_tensor_fields(fn, self)


class DcopBase(TensorDataClassMixin):
    pass


class ModuleBase[ConfigT: ConfigBase, PolicyT: PolicyBase](nn.Module, ProfileMixin, ABC):
    """Base for config- and policy-managed physical modules.

    Construct the owned module tree, place it on its device, then fabricate and
    program before execution. Construction registers fixed tensor sources with
    `_register_nonpersistent_buffer`. These buffers migrate with the module and
    are excluded from `state_dict`.

    Fabricated and programmed tensors are ordinary attributes produced by their
    lifecycle operations. They are neither migrated nor serialized with the
    module. Moving a module after materializing them requires fabrication and
    programming to run again. Per-call snapshots are local results.

    `fabricate()` invokes `_sample_fabrication_variation()` on this node before
    its registered NeuroX descendants, including those inside plain containers.
    Each hook rebuilds only its own state from nominal sources; it does not
    traverse children. Programming is dispatched explicitly by the owner.

    Stamp names after assembling the tree and before profiling. Re-stamping
    updates the subtree names; rebuilding from configuration requires a new
    fabrication and programming lifecycle.

    Args:
        inst_shape: Hardware-instance shape. Physical axes encode multiplicity;
            singleton axes may reserve positions for runtime broadcasting.
    """

    __qualified_name: str

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        if any(size <= 0 for size in inst_shape):
            raise ValueError(f"inst_shape extents must be positive; got {inst_shape}")
        self.__config = config
        self.__policy = policy
        self.__inst_shape = inst_shape

    @property
    @final
    def config(self) -> ConfigT:
        return self.__config

    @property
    @final
    def policy(self) -> PolicyT:
        return self.__policy

    @property
    @final
    def inst_shape(self) -> tuple[int, ...]:
        return self.__inst_shape

    @property
    @final
    def inst_count(self) -> int:
        return math.prod(self.__inst_shape)

    @final
    def _register_nonpersistent_buffer(self, name: str, tensor: Tensor) -> None:
        nn.Module.register_buffer(self, name, tensor, persistent=False)

    @property
    @final
    def qualified_name(self) -> str:
        """Hierarchical name the module's tree stamped onto it.

        Raises:
            RuntimeError: No tree has stamped this module yet.
        """
        try:
            return self.__qualified_name
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__} carries no name stamp; "
                "call neurox.stamp_names(model) once the model is assembled"
            ) from None

    @final
    def stamp_names(self, *, qualified_name: str = "") -> None:
        """Stamp this module and its NeuroX subtree with hierarchical names."""
        self.__qualified_name = qualified_name
        for relative_name, child in neurox_children(self):
            child_name = relative_name if not qualified_name else f"{qualified_name}.{relative_name}"
            child.stamp_names(qualified_name=child_name)

    @final
    def fabricate(self) -> None:
        """Resample static manufacturing variation across this module subtree."""
        self._sample_fabrication_variation()
        for _, child in neurox_children(self):
            child.fabricate()

    def _sample_fabrication_variation(self) -> None:
        pass


type NeuroxModule = ModuleBase[ConfigBase, PolicyBase]


def neurox_roots(model: nn.Module) -> list[NeuroxModule]:
    """Collect the outermost NeuroX modules `model` holds.

    The walk stops descending at the first `ModuleBase` it meets, so a root
    covers its own NeuroX children instead of listing them beside it. A plain
    container may hold several roots. Roots are deduplicated by identity and
    retain their first-appearance order.

    Returns:
        The outermost NeuroX modules.
    """
    if isinstance(model, ModuleBase):
        return [model]
    return [module for _, module in neurox_children(model)]


def neurox_children(
    module: nn.Module,
) -> list[tuple[str, NeuroxModule]]:
    """Collect the nearest NeuroX descendants.

    Plain module containers are traversed; descent stops at each NeuroX module.
    Repeated references are deduplicated by identity, retaining the first path
    in registered-child order.

    Returns:
        Pairs of relative registered paths and NeuroX modules.
    """
    children: list[tuple[str, NeuroxModule]] = []
    seen: set[NeuroxModule] = set()
    for name, child in module.named_children():
        if isinstance(child, ModuleBase):
            candidates = [(name, child)]
        else:
            candidates = [
                (f"{name}.{relative_name}", descendant) for relative_name, descendant in neurox_children(child)
            ]
        for relative_name, descendant in candidates:
            if descendant in seen:
                continue
            seen.add(descendant)
            children.append((relative_name, descendant))
    return children
