"""Abstract base class for TIA models.

See also:
    docs/dev/modules/analog/tia/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from neurox.common.config_dispatch import ConfigDispatchMixin
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

    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0
    latency_per_op__ns: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class TIA(nn.Module, ProfiledModule, ConfigDispatchMixin["TIAConfig", "TIA"], ABC):
    """Abstract base for transimpedance-amp clamp drivers."""

    def __init__(
        self,
        *,
        cfg: TIAConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler."""
        del cfg, T__K, dtype  # captured by the subclass init
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

    @classmethod
    def from_config(
        cls,
        *,
        cfg: TIAConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> TIA:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(cfg)
        return impl(cfg=cfg, name=name, T__K=T__K, dtype=dtype)

    @property
    @abstractmethod
    def v_ref__V(self) -> float:
        """Reference voltage [V]."""
        raise NotImplementedError
