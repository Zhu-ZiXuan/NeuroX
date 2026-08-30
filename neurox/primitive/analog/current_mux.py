"""Ideal current multiplexer — single-ended N:1 time-share transport block.

See Also:
    docs/reference/primitive/analog/current_mux.md
"""

from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class ImuxConfig(ConfigBase):
    mux_ratio: int
    """N in the N:1 ratio of inputs to each output lane."""
    mux_gain: float
    """Matched transport gain shared by every lane."""

    def validate(self) -> None:
        self._require_pos(self.mux_ratio, "mux_ratio")
        self._require_pos(self.mux_gain, "mux_gain")


class ImuxPolicy(PolicyBase):
    pass


class Imux(ModuleBase[ImuxConfig, ImuxPolicy]):
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

    def transport(self, i__uA: Tensor) -> Tensor:
        """Apply the mux transport gain elementwise.

        Args:
            i__uA: Single-ended input currents.

        Returns:
            Gained currents, one value per `i__uA` element.
        """
        return self.config.mux_gain * i__uA
