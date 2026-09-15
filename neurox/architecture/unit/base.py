"""Shared interfaces for integer compute units."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class UnitBase(ABC):
    """Value-domain and execution-metadata interfaces shared by operator families."""

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the unit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the unit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_bits(self) -> int | None:
        """ADC output width; `None` when the unit never quantizes its output."""
        raise NotImplementedError

    @abstractmethod
    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float | Tensor:
        """Return the ideal-unit codes represented by one output code.

        Args:
            quantization_mode: Index selecting the runtime quantization window.
            adc_active_bits: Active ADC resolution; `None` requests the
                unit's highest available precision.
        """
        raise NotImplementedError

    @abstractmethod
    def latency__ns(
        self,
        input_shape: tuple[int, ...],
        *,
        adc_active_bits: int | None,
    ) -> float | Tensor:
        """Latency of one complete operator call.

        The unit derives runtime-dependent serial extents from
        `input_shape`; construction fixes the remaining extents. An
        implementation whose latency is fully fixed may ignore the shape.

        Args:
            input_shape: Layout of the operand the unit's operator receives.
            adc_active_bits: Active ADC resolution; `None` requests the
                unit's highest available precision.
        """
        raise NotImplementedError
