"""Abstract base class for current-domain DAC models.

See also:
    docs/internals/primitive/analog/current_dac/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class IdacConfig(AnalogConfig, ABC):
    """Base config for current-domain DAC implementations.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IdacPolicy(AnalogPolicy, ABC):
    """Abstract marker base for current-DAC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=IdacConfig)
PolicyT = TypeVar("PolicyT", bound=IdacPolicy)


class Idac(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin["IdacConfig", "IdacPolicy", "Idac"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Base class for current-domain DAC implementations.

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
        config: IdacConfig,
        policy: IdacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Idac:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered current-DAC implementation.
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
        """Convert integer digital codes to analog output currents.

        Args:
            code: Integer input codes in ``[0, code_max]``.
                Shape: ``[...]``.

        Returns:
            Analog output current [uA], one value per ``code`` element. Dynamic
            energy and latency are emitted through the profiler side channel.
            Shape: ``[...]``.
        """
        raise NotImplementedError
