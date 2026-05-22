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

from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin


@dataclass(frozen=True, slots=True)
class AdcOperationPoint:
    """ADC operating point — the runtime selection passed per call.

    Attributes:
        adc_mode: Operating-point index, ``[0, mode_num)``.
        adc_bits: Active bit width, ``1 ≤ adc_bits ≤ max_bits``.
    """

    adc_mode: int
    adc_bits: int


@dataclass(frozen=True)
class AdcCalibrationRecord(ValidateMixin):
    """One row of the ADC ``adc_operation_point → rescale_factor`` lookup table.

    Attributes:
        adc_mode: Operating-point index.
        adc_bits: Active bit width.
        rescale_factor: Rescale factor; ``floor(M_ideal * rescale_factor) == code``.
    """

    adc_mode: int
    adc_bits: int
    rescale_factor: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.adc_mode, "adc_mode")
        self._require_nonneg(self.adc_bits, "adc_bits")


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


class ADC(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["ADCConfig"], "ADC"], ABC):
    """Abstract base class for ADC implementations."""

    @classmethod
    def from_config(
        cls,
        *,
        cfg: ADCConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> ADC:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

    def __init__(
        self,
        *,
        cfg: ADCConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler.

        Args:
            cfg: Concrete configuration dataclass.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
        """
        del cfg, dtype, T__K  # captured by the subclass init
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self._inst_shape = inst_shape

    @property
    @abstractmethod
    def mode_num(self) -> int:
        """Number of operating points the ADC supports — valid ``adc_mode``
        values lie in ``[0, mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_bits(self) -> int:
        """Physical bit width — the maximum ``adc_bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Digitise a differential analog voltage into an integer code.

        Args:
            v_pos__V: Positive-side analog input voltage [V].  Shape:
                arbitrary.
            v_neg__V: Negative-side analog input voltage [V].  Same
                shape as ``v_pos__V``.
            adc_operation_point: Runtime operating point.

        Returns:
            Integer code tensor in ``[0, 2 ** adc_operation_point.adc_bits - 1]``, same
            shape as ``v_pos__V``.  Dynamic energy and latency are
            emitted through the profiler side channel.
        """
        raise NotImplementedError

    @abstractmethod
    def latency_per_op__ns(self, *, adc_operation_point: AdcOperationPoint) -> float:
        """Return the per-conversion latency [ns].

        Args:
            adc_operation_point: Runtime operating point.

        Returns:
            Latency in [ns].
        """
        raise NotImplementedError

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
