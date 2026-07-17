"""Abstract base class for voltage-domain ADC models.

See also:
    docs/reference/primitive/analog/voltage_adc/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class VoltageAdcConfig(AnalogConfig, ABC):
    """Base config for voltage-domain ADC implementations.

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
class VoltageAdcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for voltage-ADC-family nonideality policies."""


class VoltageAdc(
    AnalogBase[VoltageAdcConfig, VoltageAdcPolicy],
    RegistryMixin[type["VoltageAdcConfig"], "VoltageAdc"],
    ABC,
):
    """Abstract base class for voltage-domain ADC implementations."""

    @classmethod
    def from_config(
        cls,
        *,
        config: VoltageAdcConfig,
        policy: VoltageAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> VoltageAdc:
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
        config: VoltageAdcConfig,
        policy: VoltageAdcPolicy,
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
    def max_bits(self) -> int:
        """Physical bit width — the maximum ``adc_bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Digitise a differential analog voltage into a signed integer code.

        Args:
            v_pos__V: Positive-side analog input voltage.  Shape:
                arbitrary.
            v_neg__V: Negative-side analog input voltage.  Same
                shape as ``v_pos__V``.
            v_refs__V: All injected reference taps, shape
                ``(*inst, num_refs)``; the impl selects one with
                ``adc_operation_point.adc_mode``. Reference-agnostic:
                supplied per call by the caller from its
                :class:`~neurox.primitive.analog.VoltageReference`.
            adc_operation_point: Runtime operating point.

        Returns:
            Signed integer code tensor, same shape as ``v_pos__V``, in
            the range reported by :meth:`signed_range` for ``adc_bits``.
            The consumer model is ``M_ideal ≈ code · rescale_factor``
            (``rescale_factor`` strictly positive). Dynamic energy and
            latency are emitted through the profiler side channel.
        """
        raise NotImplementedError

    @abstractmethod
    def signed_range(self, adc_bits: int) -> tuple[int, int]:
        """Return ``(min_code, max_code)`` the ADC can emit at ``adc_bits``.

        For ADCs whose code count matches ``2 ** adc_bits`` exactly,
        this is the canonical
        ``(-2 ** (adc_bits - 1), 2 ** (adc_bits - 1) - 1)``. For ADCs
        whose code count is **not** a power of two, the returned bounds
        reflect the actual realisable signed code range.
        """
        raise NotImplementedError
