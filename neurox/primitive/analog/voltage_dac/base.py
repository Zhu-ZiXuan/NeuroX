"""Abstract base class for voltage-domain DAC models.

See also:
    docs/internals/primitive/analog/voltage_dac/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class VdacConfig(AnalogConfig, ABC):
    """Base config for voltage-domain DAC implementations.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VdacPolicy(AnalogPolicy, ABC):
    """Abstract marker base for voltage-DAC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=VdacConfig)
PolicyT = TypeVar("PolicyT", bound=VdacPolicy)


class Vdac(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin["VdacConfig", "VdacPolicy", "Vdac"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Base class for voltage-domain DAC implementations.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        del dtype, T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @classmethod
    def from_config(
        cls,
        *,
        config: VdacConfig,
        policy: VdacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Vdac:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered voltage-DAC implementation.
        """
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
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
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes in ``[0, code_max]``.
                Shape: ``[...]``.

        Returns:
            Analog output voltage [V], at the same shape as ``code``. Dynamic
            energy is emitted through the profiler side channel.
            Shape: ``[...]``.
        """
        raise NotImplementedError
