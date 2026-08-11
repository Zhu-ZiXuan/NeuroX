"""Abstract base class for single-ended current-domain ADC models.

See also:
    docs/internals/primitive/analog/current_adc/base.md
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
class IadcObservation:
    """One :meth:`Iadc.convert` call, captured for calibration/diagnostics.

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


class IadcProber(Prober[IadcObservation]):
    """Capture current-ADC conversion observations."""

    _active_stack: ClassVar[list[Prober[IadcObservation]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[IadcObservation]]:
        return cls._active_stack


class IadcConfig(AnalogConfig, ABC):
    """Base config for single-ended current-domain ADC implementations.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IadcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for single-ended-current-ADC-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=IadcConfig)
PolicyT = TypeVar("PolicyT", bound=IadcPolicy)


class Iadc(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[
        "IadcConfig",
        "IadcPolicy",
        "Iadc",
    ],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Base class for single-ended current ADCs with injected references.

    The base owns the ``bits`` contract and nothing else about the call:
    ``bits`` is base semantics because :meth:`unsigned_range` is declared
    here. How many reference taps a conversion consumes is the concrete
    converter's own circuit property, so it is neither declared nor
    validated at this level; a ladder the leaf cannot use fails inside
    that leaf.

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
        config: IadcConfig,
        policy: IadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Iadc:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered current-ADC implementation.
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
    def max_bits(self) -> int:
        """Physical bit width — the maximum ``bits`` a ``convert`` call may request."""
        raise NotImplementedError

    def _check_bits(self, bits: int) -> None:
        """Require a resolution this converter's own bit width supports.

        Args:
            bits: Requested conversion resolution [bits].

        Raises:
            ValueError: ``bits`` is outside ``[1, max_bits]``.
        """
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"require: bits ({bits}) in [1, max_bits ({self.max_bits})]")

    @abstractmethod
    def latency__ns(self, *, bits: int) -> float:
        """Duration of one :meth:`convert` call at ``bits`` [ns].

        A conversion is the only thing a current ADC spends time on, and how
        long it lasts follows from the resolution the call executes, so the
        executed bit count is the whole question. The formula is the concrete
        converter's own — a flat comparison window, a sum over search steps —
        so the base declares no default.

        Args:
            bits: Conversion resolution [bits] in ``[1, max_bits]``.

        Returns:
            Duration of one conversion at ``bits``.
        """
        raise NotImplementedError

    def convert(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Bit width is ADC-internal: the injected ladder states the converter's
        own wiring and does not follow the requested resolution, which the
        converter realizes by running fewer decision cycles over it.

        Args:
            i_in__uA: Non-negative magnitude current.
                Shape: ``[...]``.
            i_refs__uA: Reference ladder with the taps on the last axis and the
                leading dims right-broadcasting against ``i_in__uA``. The tap
                count ``n_ref`` is the concrete converter's circuit property,
                not a base-level contract.
                Shape: ``[..., n_ref]``.
            bits: Conversion resolution [bits] in ``[1, max_bits]``.

        Returns:
            Unsigned integer code tensor, one code per ``i_in__uA`` element, in
            the range reported by :meth:`unsigned_range` for ``bits``. For a
            deterministic converter the code at ``bits`` is the code at
            ``max_bits`` right-shifted by ``max_bits - bits``. Dynamic energy
            is emitted through the profiler side channel.
            Shape: ``[...]``.

        Raises:
            ValueError: ``bits`` is outside ``[1, max_bits]``.
        """
        self._check_bits(bits)
        code = self._convert_impl(i_in__uA, i_refs__uA, bits=bits)
        if IadcProber.active():
            IadcProber.submit(
                IadcObservation(
                    i_in__uA=i_in__uA,
                    code=code,
                    bits=bits,
                ),
            )
        return code

    @abstractmethod
    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Convert inputs according to the :meth:`convert` contract."""
        raise NotImplementedError

    @abstractmethod
    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Return ``(min_code, max_code)`` the ADC can emit at ``bits``.

        For ADCs whose code count matches ``2 ** bits`` exactly, this is
        ``(0, 2 ** bits - 1)``.
        """
        raise NotImplementedError
