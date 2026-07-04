"""Abstract base class for DAC models.

See also:
    docs/reference/analog/dac/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.common.mixin import RegistryMixin


@dataclass(frozen=True)
class DACConfig(CircuitConfig):
    """Base config for DAC models."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()


@dataclass(frozen=True)
class DACPolicy:
    """Abstract marker base for DAC-family nonideality policies."""


class DAC(CircuitBase[DACConfig], RegistryMixin[type["DACConfig"], "DAC"]):
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
        del policy, dtype, T__K  # captured by the subclass init
        super().__init__(config=config, name=name, inst_shape=inst_shape)

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
