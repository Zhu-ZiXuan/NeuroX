"""Abstract base class for current-domain DAC models.

See also:
    docs/reference/primitive/analog/current_dac/README.md
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
class CurrentDacConfig(AnalogConfig, ABC):
    """Base config for current-domain DAC implementations.

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
class CurrentDacPolicy(AnalogPolicy, ABC):
    """Abstract marker base for current-DAC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=CurrentDacConfig)
PolicyT = TypeVar("PolicyT", bound=CurrentDacPolicy)


class CurrentDac(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[type["CurrentDacConfig"], "CurrentDac"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for current-domain DAC implementations.

    A current DAC drives an unsigned integer code onto a single-ended analog
    current. The DAC self-holds its own code-to-current transfer, so it has no
    per-call operating point analogous to the ADC's ``(mode, bits)``:
    :meth:`convert` takes only the code.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: CurrentDacConfig,
        policy: CurrentDacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CurrentDac:
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
        """Convert integer digital codes to analog output currents.

        Args:
            code: Integer input codes in ``[0, code_max]``. Shape: arbitrary.

        Returns:
            Analog output current [uA], same shape as ``code``. Dynamic energy
            and latency are emitted through the profiler side channel.
        """
        raise NotImplementedError
