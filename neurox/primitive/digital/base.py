"""Shared base for digital, integer-exact circuit modules.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

from abc import ABC
from typing import final

from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule


class DigitalConfig(ConfigBase, base_only=True):
    """PPA values characterize one circuit geometry at the caller's operating clock."""

    # === Arithmetic ===

    bit_width: int
    """Signed output bit width; the result wraps modulo `2^bit_width` into
    `[-2^(bit_width-1), 2^(bit_width-1) - 1]`."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DigitalPolicy(PolicyBase):
    pass


_Config = DigitalConfig
_Policy = DigitalPolicy


class DigitalBase(ProfileModule, ABC, base_only=True):
    """Base for digital, integer-exact circuit blocks.

    Operands share a caller-selected integer dtype, which arithmetic and
    results preserve. Callers choose a dtype that represents the operands,
    positional weights, wrap constants, and arithmetic intermediates.
    Operations apply `_wrap_output` to the configured output width.

    Expand the configured per-operation energy from a scalar onto the evaluated
    layout, using the operand layout before any reduction. That scalar fixes
    the energy dtype independently of the integer operands. Callers supply the
    full leading and instance extents; the block inserts no instance axes.
    """

    config: _Config
    policy: _Policy

    def __init__(self, *, config: _Config, policy: _Policy, inst_shape: tuple[int, ...]) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    # === Required by base class ===

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    # === Tools for subclass and internal use ===

    @final
    def _wrap_output(self, output: Tensor) -> Tensor:
        """Wrap integer output to signed `bit_width`, preserving shape, dtype and device."""
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        return (output + half) % full - half
