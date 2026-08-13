"""Root bases for physical modules and their config/policy dataclasses."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import Field, dataclass, field
from typing import Generic, TypeVar, dataclass_transform, final

import torch.nn as nn

from .fabricate_mixin import FabricateMixin
from .profile_mixin import ProfileMixin
from .serialize_mixin import SerializeMixin
from .validate_mixin import ValidateMixin


@dataclass_transform(frozen_default=True, kw_only_default=True, field_specifiers=(field, Field))
@dataclass(frozen=True, kw_only=True)
class ConfigBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module configurations.

    A subclass declares its fields as annotated class attributes without
    defaults, and must not apply `@dataclass` or define `__init__` or
    `__post_init__`; this base supplies a frozen, keyword-only dataclass whose
    construction ends in `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must implement validate(), not __post_init__()")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this configuration, raising `ValueError` on violation."""


@dataclass_transform(frozen_default=True, kw_only_default=True, field_specifiers=(field, Field))
@dataclass(frozen=True, kw_only=True)
class PolicyBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module runtime policies.

    A subclass declares its fields as annotated class attributes without
    defaults, and must not apply `@dataclass` or define `__init__` or
    `__post_init__`; this base supplies a frozen, keyword-only dataclass whose
    construction ends in `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        if "__post_init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must implement validate(), not __post_init__()")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the local constraints on this runtime policy, raising `ValueError` on violation."""


# ConfigT and PolicyT are covariant across every module family: a config or policy is
# produced (read-only properties, injected once at construction) and never consumed by an
# instance method. Variance constraint, for this pair and its family-level counterparts:
# instance methods must never take ConfigT or PolicyT as a parameter and must take the
# abstract base instead (`__init__` is exempt). mypy's variance check is shallow —
# `type[T]` and `list[T]` parameter positions go unflagged — so the constraint is
# partly documentation-enforced.
ConfigT = TypeVar("ConfigT", bound=ConfigBase, covariant=True)
PolicyT = TypeVar("PolicyT", bound=PolicyBase, covariant=True)


class ModuleBase(FabricateMixin, nn.Module, ProfileMixin, Generic[ConfigT, PolicyT], ABC):
    """Base for config- and policy-managed physical modules.

    A module whose PPA is owned elsewhere sets `is_profile_target = False`.

    Args:
        config: Immutable physical configuration.
        policy: Immutable runtime policy.
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
