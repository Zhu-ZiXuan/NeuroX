"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import dataclass_transform, final

import torch.nn as nn
from torch import Tensor

from .profile_mixin import ProfileMixin
from .serialize_mixin import SerializeMixin
from .tensor_dataclass import TensorDataClassBase
from .tensor_group_mixin import TensorGroupMixin
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
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must implement validate(), not __post_init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this configuration, raising `ValueError` on violation."""


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
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must implement validate(), not __post_init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this runtime policy, raising `ValueError` on violation."""


class SnapBase(TensorDataClassBase, TensorGroupMixin):
    """Transform all tensor fields together under one per-call layout."""


class DcopBase(TensorDataClassBase):
    pass


# ConfigT and PolicyT are covariant across every module family: a config or policy is
# produced (read-only properties, injected once at construction), never consumed by an
# instance method. So no instance method here or in a family-level counterpart takes one as
# a parameter; it takes the abstract base instead, `__init__` excepted. A PEP 695 type
# parameter has its variance inferred rather than declared, so only this note enforces it.
class ModuleBase[ConfigT: ConfigBase, PolicyT: PolicyBase](nn.Module, ProfileMixin, ABC):
    """Base for config- and policy-managed physical modules.

    A module whose PPA is owned elsewhere sets `is_profile_target = False`.
    A subclass owning local fabricated state overrides
    `_sample_fabrication_variation()` to rebuild that state from its nominal
    buffers. The hook handles only that node; `fabricate()` traverses the
    complete registered module tree. With no local fabricated state, retain
    the inherited no-op implementation.

    Args:
        inst_shape: Hardware-instance shape. Physical axes encode circuit
            multiplicity; axes explicitly fixed at one may instead reserve a
            broadcast position for a runtime axis.
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
        for relative_name, child in _neurox_children(self):
            child_name = relative_name if not qualified_name else f"{qualified_name}.{relative_name}"
            child.stamp_names(qualified_name=child_name)

    @final
    def fabricate(self) -> None:
        """Resample static manufacturing variation across this module subtree."""
        self._sample_fabrication_variation()
        for _, child in _neurox_children(self):
            child.fabricate()

    def _sample_fabrication_variation(self) -> None:
        pass


type NeuroxModule = ModuleBase[ConfigBase, PolicyBase]


def _neurox_roots(model: nn.Module) -> list[NeuroxModule]:
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
    return [module for _, module in _neurox_children(model)]


def _neurox_children(
    module: nn.Module,
) -> list[tuple[str, NeuroxModule]]:
    children: list[tuple[str, NeuroxModule]] = []
    seen: set[NeuroxModule] = set()
    for name, child in module.named_children():
        if isinstance(child, ModuleBase):
            candidates = [(name, child)]
        else:
            candidates = [
                (f"{name}.{relative_name}", descendant) for relative_name, descendant in _neurox_children(child)
            ]
        for relative_name, descendant in candidates:
            if descendant in seen:
                continue
            seen.add(descendant)
            children.append((relative_name, descendant))
    return children


def check_unique_neurox_bindings(model: nn.Module) -> None:
    """Check that each NeuroX module occupies one path in `model`.

    Raises:
        ValueError: One module instance is bound at two paths.
    """
    locations: dict[NeuroxModule, str] = {}
    for relative_name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, ModuleBase):
            continue
        if module in locations:
            raise ValueError(
                f"{type(module).__name__} is bound at both {locations[module]!r} and {relative_name!r}; "
                "one physical instance holds one location, so bind a separate instance per site"
            )
        locations[module] = relative_name


def fabricate(root: nn.Module) -> None:
    """Resample fabrication variation across every outermost NeuroX subtree."""
    for module in _neurox_roots(root):
        module.fabricate()


def stamp_names(model: nn.Module) -> None:
    """Stamp every NeuroX module of `model` with its hierarchical name.

    A module never knows its own name: the name is a property of the tree that
    holds it, and only a walk from a root can hand it out. Stamping again
    overwrites existing names, allowing a rewired model to be renamed.

    Raises:
        ValueError: One module instance sits at two locations of `model`.
    """
    check_unique_neurox_bindings(model)
    if isinstance(model, ModuleBase):
        model.stamp_names()
        return

    for qualified_name, root in _neurox_children(model):
        root.stamp_names(qualified_name=qualified_name)
