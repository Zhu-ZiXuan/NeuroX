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

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        enable_latency_record: Whether conversions emit latency events.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: IadcConfig,
        policy: IadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        enable_latency_record: bool = True,
    ) -> Iadc:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.
            enable_latency_record: Whether conversions emit latency events.

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
            enable_latency_record=enable_latency_record,
        )

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        enable_latency_record: bool = True,
    ) -> None:
        del dtype, T__K
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            enable_latency_record=enable_latency_record,
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

    def _check_refs(self, i_refs__uA: Tensor) -> None:
        """Require the full max-bits reference ladder.

        The ladder is a physical wiring property of the converter, so it never
        shrinks with the requested resolution: every call carries all
        ``2 ** max_bits - 1`` taps.

        Args:
            i_refs__uA: Reference ladder with the taps on the last axis.

        Raises:
            ValueError: The trailing tap count is not ``2 ** max_bits - 1``.
        """
        tap_num = (1 << self.max_bits) - 1
        n_taps = int(i_refs__uA.shape[-1])
        if n_taps != tap_num:
            raise ValueError(f"require: i_refs__uA n_taps ({n_taps}) == 2**max_bits - 1 ({tap_num})")

    def convert(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a single-ended magnitude current into an unsigned integer code.

        Bit width is ADC-internal: the full ladder stays wired whatever
        resolution is asked for, and the converter realizes ``bits`` by running
        fewer decision cycles over it.

        Args:
            i_in__uA: Non-negative magnitude current. Shape: arbitrary.
            i_refs__uA: Reference ladder, shape ``[*R, n_ref]`` with
                ``n_ref = 2 ** max_bits - 1`` taps ascending along the last
                axis; ``[*R]`` right-broadcasts against ``i_in__uA``.
            bits: Conversion resolution [bits] in ``[1, max_bits]``.

        Returns:
            Unsigned integer code tensor, same shape as ``i_in__uA``, in the
            range reported by :meth:`unsigned_range` for ``bits``. For a
            deterministic converter the code at ``bits`` is the code at
            ``max_bits`` right-shifted by ``max_bits - bits``. Dynamic energy
            and latency are emitted through the profiler side channel.

        Raises:
            ValueError: ``bits`` is outside ``[1, max_bits]``, or the ladder
                does not carry ``2 ** max_bits - 1`` taps.
        """
        self._check_bits(bits)
        self._check_refs(i_refs__uA)
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
