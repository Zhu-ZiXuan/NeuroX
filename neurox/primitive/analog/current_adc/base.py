"""Abstract base class for single-ended current-domain ADC models.

See also:
    docs/reference/primitive/analog/current_adc/README.md
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
class SingleEndedCurrentAdcObservation:
    """One :meth:`SingleEndedCurrentAdc.convert` call, captured for calibration/diagnostics.

    Attributes:
        i_in__uA: The call's input magnitude current.
        code: The call's returned unsigned integer code.
        bits: Conversion resolution [bits] (plain ``int``, not a tensor).
    """

    i_in__uA: Tensor
    code: Tensor
    bits: int

    def detach(self) -> Self:
        return replace(self, i_in__uA=self.i_in__uA.detach(), code=self.code.detach())


class SingleEndedCurrentAdcProber(Prober[SingleEndedCurrentAdcObservation]):
    """Capture point for the single-ended current ADC's conversion observation link.

    :class:`SingleEndedCurrentAdc` emits a :class:`SingleEndedCurrentAdcObservation`
    — the call's input magnitude current, returned code, and resolution — once
    per :meth:`SingleEndedCurrentAdc.convert` call when a prober is active.
    """

    _active_stack: ClassVar[list[Prober[SingleEndedCurrentAdcObservation]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[SingleEndedCurrentAdcObservation]]:
        """Return this observation link's active-prober stack."""
        return cls._active_stack


@dataclass(frozen=True)
class SingleEndedCurrentAdcConfig(AnalogConfig, ABC):
    """Base config for single-ended current-domain ADC implementations.

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
class SingleEndedCurrentAdcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for single-ended-current-ADC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=SingleEndedCurrentAdcConfig)
PolicyT = TypeVar("PolicyT", bound=SingleEndedCurrentAdcPolicy)


class SingleEndedCurrentAdc(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[type["SingleEndedCurrentAdcConfig"], "SingleEndedCurrentAdc"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for single-ended current-domain ADC implementations.

    A current ADC digitizes a single-ended magnitude current ``i_in__uA`` into an
    **unsigned** integer code. The input is a non-negative magnitude and the sign
    is handled outside the ADC by the caller. The ADC self-holds no reference and
    knows nothing of operating modes: the caller (the composing macro) has already
    selected the mode's row, so a per-instance ladder ``i_refs__uA`` (``n_ref =
    2 ** bits - 1`` taps ascending along the last axis, broadcasting against the
    input) and the resolution ``bits`` arrive per ``convert`` call directly.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: SingleEndedCurrentAdcConfig,
        policy: SingleEndedCurrentAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        record_latency: bool = True,
    ) -> SingleEndedCurrentAdc:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            record_latency=record_latency,
        )

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        record_latency: bool = True,
    ) -> None:
        """Register the instance with :class:`nn.Module`.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality enable flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.
            record_latency: Whether the ADC emits a latency event.
        """
        del dtype, T__K  # captured by the subclass init
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            record_latency=record_latency,
        )

    @property
    @abstractmethod
    def max_bits(self) -> int:
        """Physical bit width — the maximum ``bits`` a ``convert`` call may request."""
        raise NotImplementedError

    def convert(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Template method: delegates the conversion to :meth:`_convert_impl`,
        then, only when a :class:`SingleEndedCurrentAdcProber` is active, builds
        and emits the call's input, code, and resolution before returning the
        code unchanged.

        Args:
            i_in__uA: Non-negative magnitude current [uA]. Shape: arbitrary.
            i_refs__uA: Reference ladder [uA], shape ``[*R, n_ref]`` with
                ``n_ref = 2 ** bits - 1`` taps ascending along the last axis;
                ``[*R]`` right-broadcasts against ``i_in__uA``. Supplied per call
                by the caller, which has already selected the operating mode's
                row. Mode is invisible to the ADC.
            bits: Conversion resolution [bits]; drives the binary-search step
                count and must satisfy ``i_refs__uA.shape[-1] == 2 ** bits - 1``.

        Returns:
            Unsigned integer code tensor, same shape as ``i_in__uA``, in the
            range reported by :meth:`unsigned_range` for ``bits``. Dynamic
            energy and latency are emitted through the profiler side channel.
        """
        code = self._convert_impl(i_in__uA, i_refs__uA, bits=bits)
        if SingleEndedCurrentAdcProber.active():
            SingleEndedCurrentAdcProber.submit(
                SingleEndedCurrentAdcObservation(
                    i_in__uA=i_in__uA,
                    code=code,
                    bits=bits,
                ),
            )
        return code

    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
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

        For ADCs whose code count matches ``2 ** bits`` exactly, this is
        ``(0, 2 ** bits - 1)``.
        """
        raise NotImplementedError
