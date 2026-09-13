"""Shared base for digital, integer-exact circuit modules."""

from __future__ import annotations

from abc import ABC

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase


class DigitalConfig(ConfigBase, ABC):
    area_per_inst__um2: float
    """Silicon area of one fabricated instance."""
    leakage_per_inst__uW: float
    """Static leakage power of one fabricated instance."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DigitalPolicy(PolicyBase):
    pass


_Config = DigitalConfig
_Policy = DigitalPolicy


class DigitalBase(ModuleBase, ABC):
    """Base for digital, integer-exact circuit blocks.

    Operands share a caller-selected integer dtype, which arithmetic and
    results preserve. Callers choose a dtype that represents the operands,
    positional weights, wrap constants, and arithmetic intermediates.

    A block bills dynamic energy as a flat per-op lump: the config's per-op
    energy in a 0-dim constant, expanded rather than materialized onto the
    layout its operation evaluates once per element — the pre-reduction operand
    where the operation reduces. The constant, not the integer operand, fixes
    the energy dtype. The billed operand must span the block's instance
    multiplicity together with the call's leading dims at their true extents;
    everything past those dims folds away, so a block bills what it was handed
    and positions no axis of its own.
    """

    config: _Config
    policy: _Policy

    def __init__(self, *, config: _Config, policy: _Policy, inst_shape: tuple[int, ...]) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
