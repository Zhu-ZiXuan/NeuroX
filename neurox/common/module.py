"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Self, dataclass_transform, final

import torch
import torch.nn as nn
from torch import Tensor

from .dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin, map_single_tensor_fields
from .profile_mixin import ProfileMixin
from .serialize_mixin import SerializeMixin
from .validate_mixin import ValidateMixin

DEFAULT_T__K: float = 300.0
"""Initial temperature of every physical module."""


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

    # === For subclass to implement or override ===

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

    # === For subclass to implement or override ===

    def validate(self) -> None:
        """Check the local constraints on this runtime policy.

        Overrides call `super().validate()` before checking their own fields
        and raise `ValueError` when a constraint is violated.
        """


class SnapBase(TensorDataClassMixin, PyTreeDataClassMixin):
    """Registered snapshot PyTree with tensor transforms under one per-call layout."""

    # === Public API ===

    @final
    def expand(self, shape: tuple[int, ...]) -> Self:
        """Expand every tensor field with `Tensor.expand` semantics."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.expand(shape)

        return map_single_tensor_fields(fn, self)

    @final
    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Flatten every tensor field's `[start_dim, end_dim]` axes into one."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.flatten(start_dim, end_dim)

        return map_single_tensor_fields(fn, self)

    @final
    def index_select(self, dim: int, index: Tensor) -> Self:
        """Index-select one dim of every tensor field."""

        def fn(tensor: Tensor) -> Tensor:
            return tensor.index_select(dim, index)

        return map_single_tensor_fields(fn, self)


class DcopBase(TensorDataClassMixin):
    pass


class ModuleBase(nn.Module, ProfileMixin, ABC):
    """Base for config- and policy-managed physical modules.

    Construct the owned module tree, place it on its device, then fabricate and
    program before execution. Construction registers tensor sources with
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

    Temperature starts at `DEFAULT_T__K`. Construction stores it without
    invoking `_on_temperature_changed`. Temperature-dependent computations
    read `T__K` at their lifecycle or execution point. Temperature updates
    visit this node before its NeuroX descendants, including plain containers.

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
        config: ConfigBase,
        policy: PolicyBase,
        inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        if any(size <= 0 for size in inst_shape):
            raise ValueError(f"inst_shape extents must be positive; got {inst_shape}")
        self.config = config
        self.policy = policy
        self.inst_shape = inst_shape
        self._T__K = DEFAULT_T__K

    # === Public API ===

    @property
    @final
    def T__K(self) -> float:
        """Current temperature, changed through `set_temperature`."""
        return self._T__K

    @final
    @torch.no_grad()
    def set_temperature(self, T__K: float) -> None:
        """Update temperature and its direct dependencies across this subtree.

        Call between executions, outside compiled computations. Fabricated
        and programmed state remains unchanged until its next explicit
        lifecycle call. Newly attached children retain their own temperature
        until the next subtree update.

        Raises:
            ValueError: Temperature is not finite and strictly positive, or
                a subclass cannot represent a derived parameter.
        """
        if not (math.isfinite(T__K) and T__K > 0.0):
            raise ValueError(f"require: T__K ({T__K}) finite and > 0.0")
        self._T__K = T__K
        self._on_temperature_changed()
        for _, child in neurox_children(self):
            child.set_temperature(T__K)

    @final
    def stamp_names(self, *, qualified_name: str = "") -> None:
        """Stamp this module and its NeuroX subtree with hierarchical names."""
        self.__qualified_name = qualified_name
        for relative_name, child in neurox_children(self):
            child_name = relative_name if not qualified_name else f"{qualified_name}.{relative_name}"
            child.stamp_names(qualified_name=child_name)

    @final
    @torch.no_grad()
    def fabricate(self) -> None:
        """Resample static manufacturing variation across this module subtree."""
        self._sample_fabrication_variation()
        for _, child in neurox_children(self):
            child.fabricate()

    # === Required by base class ===

    @property
    @final
    def inst_count(self) -> int:
        return math.prod(self.inst_shape)

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

    # === For subclass to implement or override ===

    def _on_temperature_changed(self) -> None:
        """Refresh this node's direct dependencies from its current `T__K`.

        The default does nothing. Overrides update local scalar parameters
        and registered sources while preserving buffer device and dtype.
        They do not traverse children, sample randomness, or change fabricated
        or programmed state. Overrides extending a parent's update call it
        before updating their own dependencies.
        """

    def _sample_fabrication_variation(self) -> None:
        pass

    # === Tools for subclass and internal use ===

    @final
    def _register_nonpersistent_buffer(self, name: str, tensor: Tensor) -> None:
        nn.Module.register_buffer(self, name, tensor, persistent=False)


def neurox_roots(model: nn.Module) -> list[ModuleBase]:
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
) -> list[tuple[str, ModuleBase]]:
    """Collect the nearest NeuroX descendants.

    Plain module containers are traversed; descent stops at each NeuroX module.
    Repeated references are deduplicated by identity, retaining the first path
    in registered-child order.

    Returns:
        Pairs of relative registered paths and NeuroX modules.
    """
    children: list[tuple[str, ModuleBase]] = []
    seen: set[ModuleBase] = set()
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
