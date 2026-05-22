"""Abstract base class for TIA models.

See also:
    docs/dev/modules/analog/tia/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from neurox.common.fabricate import FabricateMixin
from neurox.common.registry_dispatch import RegistryDispatchMixin
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class TIAConfig(ValidateMixin):
    """Base configuration for TIA implementations.

    Attributes:
        v_ref__V: Reference clamp voltage [V].
        leakage_per_inst__uW: Static leakage per TIA instance [μW].
        area_per_inst__um2: Silicon area per TIA instance [μm²].
        latency_per_op__ns: Settling latency per VMM [ns].
    """

    v_ref__V: float

    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class TIA(FabricateMixin, nn.Module, ProfiledModule, RegistryDispatchMixin[type["TIAConfig"], "TIA"], ABC):
    """Abstract base for transimpedance-amp clamp drivers."""

    def __init__(
        self,
        *,
        cfg: TIAConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler."""
        del cfg, dtype, T__K  # captured by the subclass init
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self._inst_shape = inst_shape

    @classmethod
    def from_config(
        cls,
        *,
        cfg: TIAConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> TIA:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

    @property
    @abstractmethod
    def v_ref__V(self) -> float:
        """Reference voltage [V]."""
        raise NotImplementedError
