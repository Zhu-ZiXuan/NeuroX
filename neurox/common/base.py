"""Root bases for physical modules and their config/policy dataclasses."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar, dataclass_transform, final

import torch.nn as nn

from neurox.common.mixin import FabricateMixin, ProfileMixin, SerializeMixin, ValidateMixin


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class ConfigBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module configurations.

    Subclass requirements:
        - Declare fields as annotated class attributes without defaults.
        - Do not apply ``@dataclass`` or define ``__init__`` or
          ``__post_init__``; this base supplies a frozen, keyword-only
          dataclass.
        - Override :meth:`validate` for local constraints.
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
        """Validate this configuration."""


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class PolicyBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module runtime policies.

    Subclass requirements:
        - Declare fields as annotated class attributes without defaults.
        - Do not apply ``@dataclass`` or define ``__init__`` or
          ``__post_init__``; this base supplies a frozen, keyword-only
          dataclass.
        - Override :meth:`validate` for local constraints.
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
        """Validate this runtime policy."""


ConfigT = TypeVar("ConfigT", bound=ConfigBase)
PolicyT = TypeVar("PolicyT", bound=PolicyBase)


class ModuleBase(FabricateMixin, nn.Module, ProfileMixin, Generic[ConfigT, PolicyT], ABC):
    """Base for config- and policy-managed physical modules.

    Subclass requirements:
        - Implement ``_sample_fabricate_mismatch`` for local static state; a
          container with no local mismatch implements an explicit no-op.
        - A profile target must initialize ``_area_per_inst__um2`` and
          ``_leakage_per_inst__uW``. A module whose PPA is owned elsewhere sets
          ``is_profile_target = False``.

    Args:
        config: Immutable physical configuration.
        policy: Immutable runtime policy.
        inst_shape: Multiplicity of parallel physical instances.
        enable_latency_record: Whether this module emits latency events.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        enable_latency_record: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.__config = config
        self.__policy = policy
        self.__inst_shape = inst_shape
        self.__inst_count = math.prod(inst_shape)
        self.enable_latency_record = enable_latency_record

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
