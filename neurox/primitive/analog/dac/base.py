"""Abstract base class for DAC models.

See also:
    docs/reference/primitive/analog/dac/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class DACConfig(AnalogConfig):
    """Base config for DAC models.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
class DACPolicy(AnalogPolicy):
    """Abstract marker base for DAC-family nonideality policies."""


class DAC(AnalogBase[DACConfig, DACPolicy], RegistryMixin[type["DACConfig"], "DAC"]):
    """Abstract base class for DAC models."""

    def __init__(
        self,
        *,
        config: DACConfig,
        policy: DACPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler."""
        del dtype, T__K  # captured by the subclass init
        super().__init__(config=config, policy=policy, name=name, inst_shape=inst_shape)

    @classmethod
    def from_config(
        cls,
        *,
        config: DACConfig,
        policy: DACPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> DAC:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

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
