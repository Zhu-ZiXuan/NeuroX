"""Abstract base class for DAC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.registry_dispatch import RegistryDispatchMixin
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class DACConfig(ValidateMixin):
    """Base config for DAC models."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        pass


class DAC(nn.Module, ProfiledModule, RegistryDispatchMixin[type["DACConfig"], "DAC"], ABC):
    """Abstract base class for DAC models."""

    def __init__(
        self,
        *,
        cfg: DACConfig,
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
        cfg: DACConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> DAC:
        """Build the concrete DAC model for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, T__K=T__K, dtype=dtype)

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Circuit area per instance in [um2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Circuit leakage power per instance in [uW]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def latency_per_op__ns(self) -> float:
        """Latency per operation in [ns]."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog signals.

        Args:
            code: Integer input codes. Shape: arbitrary.

        Returns:
            Float analog signal of the same shape as ``code``.
        """
        raise NotImplementedError

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Default no-op; DAC has no static fabrication state today."""
        raise NotImplementedError
