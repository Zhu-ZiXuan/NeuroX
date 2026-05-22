"""Abstract base class for DAC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin


@dataclass(frozen=True)
class DACConfig(ValidateMixin):
    """Base config for DAC models."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        pass


class DAC(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["DACConfig"], "DAC"], ABC):
    """Abstract base class for DAC models."""

    def __init__(
        self,
        *,
        cfg: DACConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler."""
        del cfg, dtype, T__K  # captured by the subclass init
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self._inst_shape = inst_shape

    @classmethod
    def from_config(
        cls,
        *,
        cfg: DACConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> DAC:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def code_max(self) -> int:
        """Maximum valid input code (inclusive); valid codes lie in ``[0, code_max]``."""
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
