"""Abstract base class for differential voltage-domain ADC models.

See also:
    docs/internals/primitive/analog/diff_voltage_adc/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase, RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class DiffVadcRecord(RecordBase):
    """One :meth:`DiffVadc.convert` call, captured for calibration/diagnostics.

    The record covers the conversion event alone — the signal path, the
    output code, and the active bit width. References are calibrated
    constants rather than measured quantities, so they stay out of it, and
    which instance converted is the collecting caller's own knowledge.

    Attributes:
        v_pos__V: The call's positive-side input voltage.
        v_neg__V: The call's negative-side input voltage.
        code: The call's raw unsigned integer code.
        bits: Active bit width (plain ``int``, not a tensor).
    """

    v_pos__V: Tensor
    v_neg__V: Tensor
    code: Tensor
    bits: int


class DiffVadcProber(RecorderBase[DiffVadcRecord]):
    """Capture differential-voltage ADC conversion records."""


class DiffVadcConfig(AnalogConfig, ABC):
    """Base config for differential voltage-domain ADC implementations.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DiffVadcPolicy(AnalogPolicy, ABC):
    """Abstract marker base for differential-voltage-ADC nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=DiffVadcConfig, covariant=True)
PolicyT = TypeVar("PolicyT", bound=DiffVadcPolicy, covariant=True)


class DiffVadc(
    AnalogBase[ConfigT, PolicyT],
    RegistryMixin[
        "DiffVadcConfig",
        "DiffVadcPolicy",
        "DiffVadc[DiffVadcConfig, DiffVadcPolicy]",
    ],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Base class for differential voltage-domain ADC implementations.

    A converter owns its transfer structure and never its reference values:
    every ``convert`` call carries the taps in. How many taps a call needs
    is the concrete converter's own circuit property, so the base validates
    no tap count.

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
        config: DiffVadcConfig,
        policy: DiffVadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> DiffVadc[DiffVadcConfig, DiffVadcPolicy]:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered voltage-ADC implementation.
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
        """Physical bit width — the maximum ``bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def latency__ns(self, *, bits: int) -> float:
        """Duration of one :meth:`convert` call at ``bits`` [ns].

        A conversion is the only thing a differential voltage ADC spends time
        on, and how long it lasts follows from the resolution the call
        executes, so the executed bit count is the whole question. The formula
        is the concrete converter's own — a flat comparison window, a cycle
        count that grows with the resolution — so the base declares no
        default.

        Args:
            bits: Active conversion resolution [bits].

        Returns:
            Duration of one conversion at ``bits``.
        """
        raise NotImplementedError

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Digitise a differential analog voltage into a raw unsigned code.

        Args:
            v_pos__V: Positive-side analog input voltage.
                Shape: ``[...]``.
            v_neg__V: Negative-side analog input voltage, at the same
                shape as ``v_pos__V``.
                Shape: ``[...]``.
            v_refs__V: Injected reference taps, with the taps on the last
                axis. The tap count ``n_ref`` is the concrete converter's
                circuit property, not a base-level contract.
                Shape: ``[..., n_ref]``.
            bits: Active conversion resolution [bits].

        Returns:
            Raw unsigned integer code tensor, at the same shape as
            ``v_pos__V``, in the range reported by :meth:`unsigned_range` for
            ``bits``. For offset-binary codes, recover the signed value as
            ``M_ideal ≈ (code − zero_offset(bits)) · rescale_factor``
            with a positive ``rescale_factor``.
            Shape: ``[...]``.
        """
        code = self._convert_impl(
            v_pos__V,
            v_neg__V,
            v_refs__V=v_refs__V,
            bits=bits,
        )
        if DiffVadcProber.active():
            DiffVadcProber.submit(
                DiffVadcRecord(
                    v_pos__V=v_pos__V,
                    v_neg__V=v_neg__V,
                    code=code,
                    bits=bits,
                ),
            )
        return code

    @abstractmethod
    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Convert inputs according to the :meth:`convert` contract."""
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

        Subtract this offset before scaling:
        ``M_ideal ≈ (code − zero_offset(bits)) · rescale_factor``. Sign
        and offset are not folded into the emitted code. For a symmetric
        power-of-two design this is ``2 ** (bits - 1)``.
        """
        raise NotImplementedError
