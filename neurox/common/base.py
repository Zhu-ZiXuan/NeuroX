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
    """Root of every physical module.

    A subclass parameterizes the config / policy pair with its own types, so
    ``self.config`` and ``self.policy`` read back at those types. ``__init__``
    registers the node with ``nn.Module`` and binds that pair plus
    ``inst_shape``, the host state ``FabricateMixin`` and ``ProfileMixin`` read.
    ``inst_shape`` is the per-instance fabrication multiplicity — the shape a
    subclass samples its static mismatch over — and ``inst_count`` its product:
    the copies fabricated in parallel behind one module, never a serial-op
    count.

    Neither inherited surface is opt-in: every module joins the pre-order
    ``fabricate()`` cascade and is a profiling host.

    Subclass requirements:
        - Implement ``_sample_fabricate_mismatch``, the per-layer sampling step
          the inherited cascade drives.
        - Set the bare ``_area_per_inst__um2`` / ``_leakage_per_inst__uW`` in
          ``__init__``, which ``ProfileMixin`` aggregates into the reported
          ``area__um2`` / ``leakage__uW``. A module whose silicon rolls up into
          an owner's budget sets neither and overrides ``reports_static_ppa`` to
          ``False``, keeping it out of the profiler's static walk.
    """

    config: ConfigT
    policy: PolicyT

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.policy = policy
        self._inst_shape = inst_shape

    @property
    def inst_shape(self) -> tuple[int, ...]:
        return self._inst_shape

    @property
    def inst_count(self) -> int:
        return math.prod(self._inst_shape)
