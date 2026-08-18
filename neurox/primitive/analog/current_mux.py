"""Ideal current multiplexer — single-ended N:1 time-share transport block.

See Also:
    docs/reference/primitive/analog/current_mux.md
"""

from typing import ClassVar

import torch
from torch import Tensor

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class ImuxConfig(AnalogConfig):
    """Immutable configuration for `Imux`."""

    mux_ratio: int
    """N in the N:1 ratio of inputs to each output lane."""
    mux_gain: float
    """Matched transport gain shared by every lane."""

    def validate(self) -> None:
        self._require_pos(self.mux_ratio, "mux_ratio")
        self._require_pos(self.mux_gain, "mux_gain")


class ImuxPolicy(AnalogPolicy):
    """Abstract marker for Imux nonideality policy — no sources."""


class Imux(AnalogBase[ImuxConfig, ImuxPolicy]):
    """Ideal N:1 time-share current mux — identity·gain transport."""

    is_profile_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: ImuxConfig,
        policy: ImuxPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    def transport(self, i__uA: Tensor) -> Tensor:
        """Transport currents already scheduled across mux accesses and lanes.

        Args:
            i__uA: Single-ended input currents, where `access_num` equals
                `mux_ratio`.
                Shape: `[..., access_num, lane_num]`.

        Returns:
            Gained currents, one value per `i__uA` element.
            Shape: `[..., access_num, lane_num]`.

        Raises:
            ValueError: The trailing axes are not `(mux_ratio, lane_num)`.
        """
        lane_num = self.inst_shape[-1] if self.inst_shape else 1
        expected_trailing = (self.config.mux_ratio, lane_num)
        if i__uA.shape[-2:] != expected_trailing:
            raise ValueError(
                f"trailing axes must be (access_num={self.config.mux_ratio}, lane_num={lane_num}); "
                f"got {tuple(i__uA.shape[-2:])}"
            )
        return self.config.mux_gain * i__uA
