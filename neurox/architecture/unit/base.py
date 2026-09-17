"""Shared interfaces for integer compute units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from torch import Tensor

from neurox.common.module import ConfigBase, ProfileModule


class UnitConfig(ConfigBase, ABC):
    """Unit-local peripheral costs, excluding independently profiled child circuits."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


_Config = UnitConfig


class UnitBase(ProfileModule, ABC):
    """Profiled integer operators with shared value-domain and execution metadata.

    Every implementation, including an ideal operator, retains the unit's
    local static-cost contract. Child circuits report their costs separately.
    """

    config: _Config

    # === Required by base class ===

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    # === For subclass to implement or override ===

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
