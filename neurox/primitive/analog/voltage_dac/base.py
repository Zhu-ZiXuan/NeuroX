"""Abstract base class for voltage-domain DAC models.

See Also:
    docs/reference/primitive/analog/voltage_dac/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.common import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class VdacConfig(AnalogConfig, ABC):
    """Base config for voltage-domain DAC implementations."""

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VdacPolicy(AnalogPolicy, ABC):
    """Abstract marker base for voltage-DAC-family nonideality policies."""


class Vdac[ConfigT: VdacConfig, PolicyT: VdacPolicy](
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin["VdacConfig", "VdacPolicy", "Vdac[VdacConfig, VdacPolicy]"],
    ABC,
):
    """Base class for voltage-domain DAC implementations.

    A converter reports no duration: it settles inside the window the owner
    scheduling its conversions already spans, and that owner times it.
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
    ) -> Vdac[VdacConfig, VdacPolicy]:
        """Build the implementation registered for the config-policy pair.

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
        """Largest code the converter accepts; valid codes lie in `[0, code_max]`."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes in `[0, code_max]`.
                Shape: `[...]`.

        Returns:
            Analog output voltage [V], one value per `code` element. Dynamic
            energy is emitted through the profiler side channel.
            Shape: `[...]`.
        """
        raise NotImplementedError
