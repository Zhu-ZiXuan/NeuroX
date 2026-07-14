"""Root bases for physical modules and their config/policy dataclasses.

See also:
    docs/internals/common/base.md
"""

from __future__ import annotations

import math
from typing import Generic, TypeVar

import torch.nn as nn

from neurox.common.mixin import FabricateMixin, ProfileMixin, SerializeMixin, ValidateMixin


class ConfigBase(SerializeMixin, ValidateMixin):
    """Root of every configuration dataclass.

    A configuration is a frozen dataclass inheriting this root, directly or
    through a family base.
    """


class PolicyBase(SerializeMixin):
    """Root of every policy dataclass.

    A policy is a frozen dataclass of non-ideality switches.
    """


ConfigT = TypeVar("ConfigT", bound=ConfigBase)
PolicyT = TypeVar("PolicyT", bound=PolicyBase)


class ModuleBase(FabricateMixin, nn.Module, ProfileMixin, Generic[ConfigT, PolicyT]):
    config: ConfigT
    policy: PolicyT

    def __init__(
        self, *, config: ConfigT, policy: PolicyT,
        name: str = "", inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.policy = policy
        self._neurox_name = name
        self._inst_shape = inst_shape

    @property
    def inst_shape(self) -> tuple[int, ...]:
        return self._inst_shape

    @property
    def inst_count(self) -> int:
        return math.prod(self._inst_shape)
