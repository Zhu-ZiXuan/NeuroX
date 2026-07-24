"""Root bases for physical modules and their config/policy dataclasses.

See also:
    docs/internals/config_and_policy.md
"""

from __future__ import annotations

import math
from abc import ABC
from typing import Generic, TypeVar, final

import torch.nn as nn

from neurox.common.mixin import FabricateMixin, ProfileMixin, SerializeMixin, ValidateMixin


class ConfigBase(SerializeMixin, ValidateMixin, ABC):
    """Root of every configuration dataclass.

    A configuration is a frozen dataclass inheriting this root, directly or
    through a family base.
    """


class PolicyBase(SerializeMixin, ABC):
    """Root of every policy dataclass.

    A policy is a frozen dataclass of non-ideality switches.
    """


ConfigT = TypeVar("ConfigT", bound=ConfigBase)
PolicyT = TypeVar("PolicyT", bound=PolicyBase)


class ModuleBase(FabricateMixin, nn.Module, ProfileMixin, Generic[ConfigT, PolicyT], ABC):
    """Root of every physical module.

    A subclass parameterizes the config / policy pair with its own types, so
    ``self.config`` and ``self.policy`` read back at those types. ``__init__``
    registers the node with ``nn.Module`` and binds that pair plus
    ``inst_shape``, the host state ``FabricateMixin`` and ``ProfileMixin`` read.
    ``inst_shape`` is the per-instance fabrication multiplicity — the shape a
    subclass samples its static mismatch over — and ``inst_count`` its product:
    the copies fabricated in parallel behind one module, never a serial-op
    count.

    ``record_latency`` gates whether the module emits latency events; an owner
    that already bills the serial-op latency downstream passes ``False`` so the
    submodule contributes energy without double-counting time. The flag is the
    reusable home of the decision — each emitter guards its own
    ``_log_latency`` call on it.

    No inherited surface is opt-in: every module joins the pre-order
    ``fabricate()`` cascade and is a profiling host. Probe emission is not
    universal and carries no inherited surface — only the leaves that own an
    observation link emit, guarding each call on the link's ``Prober``
    subclass.

    Subclass requirements:
        - Implement ``_sample_fabricate_mismatch``, the per-layer sampling step
          the inherited cascade drives.
        - Set the bare ``_area_per_inst__um2`` / ``_leakage_per_inst__uW`` in
          ``__init__``, which ``ProfileMixin`` aggregates into the reported
          ``area__um2`` / ``leakage__uW``. A module whose silicon rolls up into
          an owner's budget sets neither and overrides ``is_profile_target`` to
          ``False``, keeping it out of the profiler's static walk.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        record_latency: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.__config = config
        self.__policy = policy
        self.__inst_shape = inst_shape
        self.__inst_count = math.prod(inst_shape)
        self.record_latency = record_latency

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
