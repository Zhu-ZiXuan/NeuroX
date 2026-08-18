"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import dataclass_transform, final

import torch.nn as nn

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
    """Base for the uniformly shaped per-call state returned by `snapshot()`."""


class DcopBase(TensorDataClassBase):
    """Base for the DC operating point returned by `solve_dc()`."""


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
        inst_shape: Multiplicity of parallel physical instances.
    """

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
        self.__inst_count = math.prod(inst_shape)

    @final
    def fabricate(self) -> None:
        """Resample static manufacturing variation across this module subtree."""
        fabricate(self)

    def _sample_fabrication_variation(self) -> None:
        pass

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
        return self.__inst_count


def fabricate(root: nn.Module) -> None:
    """Resample static manufacturing variation across a registered module tree."""
    for module in root.modules():
        if isinstance(module, ModuleBase):
            module._sample_fabrication_variation()  # noqa: SLF001  the dispatcher owns this hook
