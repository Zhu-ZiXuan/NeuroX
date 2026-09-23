"""Shared interfaces for integer compute units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule


class UnitConfig(ConfigBase, base_only=True):
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


class UnitBase(ProfileModule, ABC, base_only=True):
    """Single-instance integer operators with value-domain and execution metadata.

    Before profiling, bind the assembled subtree's profile leading rank to the
    number of independent operation axes in its input. Public operator entries
    check this rank before executing their children. Numerical execution never
    changes the configured rank; aggregation across operations belongs to callers.
    """

    config: _Config

    def __init__(
        self,
        *,
        config: _Config,
        policy: PolicyBase,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())

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
        adc_active_bits: int | None = None,
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
        adc_active_bits: int | None = None,
    ) -> float:
        """Duration of one basic operation, shared by all caller batch positions.

        Leading batch and token extents do not multiply this value. Public
        execution methods submit this scalar unchanged. The model owner
        configures profiling so every retained energy element corresponds to
        one basic operation.

        `input_shape` supplies internal runtime extents; construction fixes
        the remaining geometry. Timing is available before programming or
        numerical execution. Implementations compose internal durations before
        returning a scalar; extracting a value from a device tensor may synchronize
        that device.

        Args:
            input_shape: Layout of the operand the unit's operator receives.
            adc_active_bits: Active ADC resolution; `None` requests the
                unit's highest available precision.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _check_profile_leading_rank(self, expected_rank: int) -> None:
        if self._profile_leading_rank != expected_rank:
            raise ValueError(
                f"profiling requires profile_leading_rank={expected_rank}; got {self._profile_leading_rank}. "
                "Call set_profile_leading_rank() on the assembled unit before profiling."
            )
