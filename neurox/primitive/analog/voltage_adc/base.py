"""Abstract base class for voltage-domain ADC models.

See also:
    docs/internals/primitive/analog/voltage_adc/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar, Generic, Self, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.common.prober import Prober
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class DifferentialVoltageAdcObservation:
    """One :meth:`DifferentialVoltageAdc.convert` call, captured for calibration/diagnostics.

    Attributes:
        v_pos__V: The call's positive-side input voltage.
        v_neg__V: The call's negative-side input voltage.
        v_ref__V: The call's owner-preselected single reference tap.
        code: The call's returned raw unsigned integer code (offset-binary /
            bucket index; the zero point is recovered consumer-side).
        bits: Active bit width (plain ``int``, not a tensor).
    """

    v_pos__V: Tensor
    v_neg__V: Tensor
    v_ref__V: Tensor
    code: Tensor
    bits: int

    def detach(self) -> Self:
        return replace(
            self,
            v_pos__V=self.v_pos__V.detach(),
            v_neg__V=self.v_neg__V.detach(),
            v_ref__V=self.v_ref__V.detach(),
            code=self.code.detach(),
        )


class DifferentialVoltageAdcProber(Prober[DifferentialVoltageAdcObservation]):
    """Capture point for the voltage ADC's conversion observation link.

    :class:`DifferentialVoltageAdc` emits a :class:`DifferentialVoltageAdcObservation` — the
    call's differential input voltages, selected reference tap, returned code, and
    resolution — once per :meth:`DifferentialVoltageAdc.convert` call when a prober
    is active.
    """

    _active_stack: ClassVar[list[Prober[DifferentialVoltageAdcObservation]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[DifferentialVoltageAdcObservation]]:
        """Return this observation link's active-prober stack."""
        return cls._active_stack


@dataclass(frozen=True)
class DifferentialVoltageAdcConfig(AnalogConfig, ABC):
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
class DifferentialVoltageAdcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for voltage-ADC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=DifferentialVoltageAdcConfig)
PolicyT = TypeVar("PolicyT", bound=DifferentialVoltageAdcPolicy)


class DifferentialVoltageAdc(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[type["DifferentialVoltageAdcConfig"], "DifferentialVoltageAdc"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for voltage-domain ADC implementations."""

    @classmethod
    def from_config(
        cls,
        *,
        config: DifferentialVoltageAdcConfig,
        policy: DifferentialVoltageAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> DifferentialVoltageAdc:
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
    def max_bits(self) -> int:
        """Physical bit width — the maximum ``bits`` value."""
        raise NotImplementedError

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_ref__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into a raw unsigned code.

        Template method: delegates the conversion to :meth:`_convert_impl`,
        then, only when a :class:`DifferentialVoltageAdcProber` is active, builds and
        emits the call's inputs, code, and resolution before returning
        the code unchanged.

        Args:
            v_pos__V: Positive-side analog input voltage.  Shape:
                arbitrary.
            v_neg__V: Negative-side analog input voltage.  Same
                shape as ``v_pos__V``.
            v_ref__V: The single reference tap the owner has already
                selected, shape ``(*inst,)`` (the result of
                ``v_refs__V[..., mode]``). The ADC is reference-consuming
                but mode-blind: mode selection happens caller-side.
            bits: Active conversion resolution [bits].

        Returns:
            Raw unsigned integer code tensor, same shape as ``v_pos__V``,
            in the range reported by :meth:`unsigned_range` for
            ``bits`` (offset-binary / bucket index — the ADC does NOT
            fold the zero point in). The consumer recovers the signed
            magnitude affinely as
            ``M_ideal ≈ (code − zero_offset(bits)) · rescale_factor``
            (``rescale_factor`` strictly positive), where the zero point
            comes from :meth:`zero_offset`. Dynamic energy and latency are
            emitted through the profiler side channel.
        """
        code = self._convert_impl(
            v_pos__V,
            v_neg__V,
            v_ref__V=v_ref__V,
            bits=bits,
        )
        if DifferentialVoltageAdcProber.active():
            DifferentialVoltageAdcProber.submit(
                DifferentialVoltageAdcObservation(
                    v_pos__V=v_pos__V,
                    v_neg__V=v_neg__V,
                    v_ref__V=v_ref__V,
                    code=code,
                    bits=bits,
                ),
            )
        return code

    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_ref__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Conversion body a concrete impl provides; contract as :meth:`convert`.

        Deliberately ``NotImplementedError``-raising rather than
        ``@abstractmethod``: a capture-style subclass may override
        :meth:`convert` wholesale and must stay instantiable without a
        conversion body.
        """
        raise NotImplementedError

    @abstractmethod
    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Return ``(min_code, max_code)`` the ADC can emit at ``bits``.

        The code is raw (unsigned / offset-binary), so ``min_code`` is
        ``0``. For ADCs whose code count matches ``2 ** bits`` exactly
        this is ``(0, 2 ** bits - 1)``; for ADCs whose code count is
        **not** a power of two the upper bound reflects the actual
        realisable code count.
        """
        raise NotImplementedError

    @abstractmethod
    def zero_offset(self, bits: int) -> int:
        """Return the raw code representing analog zero at ``bits``.

        The consumer subtracts this offset before scaling:
        ``M_ideal ≈ (code − zero_offset(bits)) · rescale_factor``. Sign
        and offset handling live entirely on the consumer side — the ADC
        emits only the raw unsigned code. For a symmetric power-of-two
        design this is ``2 ** (bits - 1)``; an asymmetric or
        single-ended design places it elsewhere.
        """
        raise NotImplementedError
