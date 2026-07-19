"""Abstract base class for current-domain ADC models.

See also:
    docs/reference/primitive/analog/current_adc/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.common.prober import AdcProber
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class CurrentAdcConfig(AnalogConfig, ABC):
    """Base config for current-domain ADC implementations.

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
class CurrentAdcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for current-ADC-family nonideality policies."""


class CurrentAdc(
    AnalogBase[CurrentAdcConfig, CurrentAdcPolicy],
    RegistryMixin[type["CurrentAdcConfig"], "CurrentAdc"],
    ABC,
):
    """Abstract base class for current-domain ADC implementations.

    A current ADC digitizes a single-ended magnitude current ``i_in__uA`` into an
    **unsigned** integer code. The input is a non-negative magnitude and the sign
    is handled outside the ADC by the caller. The ADC self-holds no reference; the
    per-call :class:`AdcOperationPoint` selects the runtime operating point.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: CurrentAdcConfig,
        policy: CurrentAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CurrentAdc:
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
        config: CurrentAdcConfig,
        policy: CurrentAdcPolicy,
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

    def convert(self, i_in__uA: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Template method: delegates the conversion to :meth:`_convert_impl`,
        then emits the call's input, code, and operating point on the
        ``adc.convert`` probe channel (a no-op without an active
        :class:`~neurox.common.prober.Prober`) before returning the code
        unchanged.

        Args:
            i_in__uA: Non-negative magnitude current [uA]. Shape: arbitrary.
            adc_operation_point: Runtime operating point.

        Returns:
            Unsigned integer code tensor, same shape as ``i_in__uA``, in the
            range reported by :meth:`unsigned_range` for ``adc_bits``. Dynamic
            energy and latency are emitted through the profiler side channel.
        """
        code = self._convert_impl(i_in__uA, adc_operation_point=adc_operation_point)
        self._probe_record(
            AdcProber.ADC_CONVERT,
            i_in__uA=i_in__uA,
            code=code,
            adc_mode=torch.tensor(adc_operation_point.adc_mode),
            adc_bits=torch.tensor(adc_operation_point.adc_bits),
        )
        return code

    def _convert_impl(self, i_in__uA: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Conversion body a concrete impl provides; contract as :meth:`convert`.

        Deliberately ``NotImplementedError``-raising rather than
        ``@abstractmethod``: a capture-style subclass may override
        :meth:`convert` wholesale and must stay instantiable without a
        conversion body.
        """
        raise NotImplementedError

    @abstractmethod
    def unsigned_range(self, adc_bits: int) -> tuple[int, int]:
        """Return ``(min_code, max_code)`` the ADC can emit at ``adc_bits``.

        For ADCs whose code count matches ``2 ** adc_bits`` exactly, this is
        ``(0, 2 ** adc_bits - 1)``.
        """
        raise NotImplementedError
