"""Abstract base class for ADC models.

See also:
    docs/dev/modules/analog/adc/README.md
"""

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
class ADCConfig(ValidateMixin):
    """Base config for ADC implementations."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class ADCMode(ValidateMixin):
    """One operating mode of a multi-mode ADC.

    Attributes:
        n_bits: Bit width of the mode.
        n_states: Number of analog states represented by the mode.
        max_signal: Full-scale differential signal [V].
    """

    n_bits: int
    n_states: int
    max_signal: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"require: n_bits ({self.n_bits}) >= 1")
        if self.n_states < 2:
            raise ValueError(f"require: n_states ({self.n_states}) >= 2")
        if self.n_states > (1 << self.n_bits):
            raise ValueError(f"require: n_states ({self.n_states}) <= 2**n_bits ({1 << self.n_bits})")
        if not (self.max_signal > 0.0):
            raise ValueError(f"require: max_signal ({self.max_signal}) > 0")

    @property
    def n_codes(self) -> int:
        """Number of distinct output codes — ``2 ** n_bits``."""
        return 1 << self.n_bits

    @property
    def lsb(self) -> float:
        """Bin width — ``max_signal / n_codes``."""
        return self.max_signal / self.n_codes


class ADC(nn.Module, ProfiledModule, RegistryDispatchMixin[type["ADCConfig"], "ADC"], ABC):
    """Abstract base class for ADC implementations."""

    @classmethod
    def from_config(
        cls,
        *,
        cfg: ADCConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> ADC:
        """Build the concrete ADC implementation for `type(cfg)`.

        Args:
            cfg: ADC configuration.
            name: Profiler/debug name.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.

        Returns:
            Concrete ADC implementation registered for `type(cfg)`.
        """
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, T__K=T__K, dtype=dtype)

    def __init__(
        self,
        *,
        cfg: ADCConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler.

        Args:
            cfg: ADC configuration.
            name: Profiler/debug name.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
        """
        del cfg, T__K, dtype  # captured by the subclass init
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

    @abstractmethod
    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        mode: int,
        bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into an integer code.

        Args:
            v_pos__V: Positive-side analog input voltage [V].  Shape:
                arbitrary.
            v_neg__V: Negative-side analog input voltage [V].  Same
                shape as ``v_pos__V``.
            mode: Runtime operating-point index.  ``[0, n_modes)``.
            bits: Active bit width for this conversion.

        Returns:
            Integer code tensor in ``[0, 2 ** bits - 1]``, same shape
            as ``v_pos__V``.  Dynamic energy and latency are emitted
            through the profiler side channel.
        """
        raise NotImplementedError

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        raise NotImplementedError

    @abstractmethod
    def latency_per_op__ns(self, *, bits: int) -> float:
        """Return the per-conversion latency [ns].

        Args:
            bits: Active bit width.

        Returns:
            Latency in [ns].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Silicon area per ADC instance in [um^2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Static leakage power per ADC instance in [uW]."""
        raise NotImplementedError
