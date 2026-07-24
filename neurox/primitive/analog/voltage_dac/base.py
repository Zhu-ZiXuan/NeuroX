"""Abstract base class for voltage-domain DAC models.

See also:
    docs/reference/primitive/analog/voltage_dac/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class VoltageDacConfig(AnalogConfig, ABC):
    """Base config for voltage-domain DAC implementations.

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
class VoltageDacPolicy(AnalogPolicy, ABC):
    """Abstract marker base for voltage-DAC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=VoltageDacConfig)
PolicyT = TypeVar("PolicyT", bound=VoltageDacPolicy)


class VoltageDac(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[type["VoltageDacConfig"], "VoltageDac"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for voltage-domain DAC implementations.

    A voltage DAC drives an unsigned integer code onto an analog voltage. The
    DAC self-holds its own code-to-voltage transfer, so it has no per-call
    operating point analogous to the ADC's ``(mode, bits)``: :meth:`convert`
    takes only the code.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: VoltageDacConfig,
        policy: VoltageDacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> VoltageDac:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module`.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality enable flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.
        """
        del dtype, T__K  # captured by the subclass init
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    @abstractmethod
    def code_max(self) -> int:
        """Maximum valid input code (inclusive); valid codes lie in ``[0, code_max]``."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes in ``[0, code_max]``. Shape: arbitrary.

        Returns:
            Analog output voltage [V], same shape as ``code``. Dynamic energy
            and latency are emitted through the profiler side channel.
        """
        raise NotImplementedError
