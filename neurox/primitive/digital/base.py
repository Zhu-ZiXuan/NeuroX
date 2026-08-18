"""Shared base for digital, integer-exact circuit modules."""

from __future__ import annotations

from abc import ABC

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class DigitalConfig(ConfigBase, ABC):
    """Static PPA fields shared by every digital, integer-exact block."""

    area_per_inst__um2: float
    """Silicon area of one fabricated instance."""
    leakage_per_inst__uW: float
    """Static leakage power of one fabricated instance."""

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DigitalPolicy(PolicyBase):
    """Empty policy marker — integer-exact blocks carry no nonidealities."""


class DigitalBase[ConfigT: DigitalConfig](ModuleBase[ConfigT, DigitalPolicy], ABC):
    """Base for digital, integer-exact circuit blocks.

    A block bills dynamic energy as a flat per-op lump: the config's per-op
    energy in a 0-dim constant, expanded rather than materialized onto the
    layout its operation evaluates once per element — the pre-reduction operand
    where the operation reduces. The constant, not the integer operand, fixes
    the energy dtype. The billed operand must span the block's instance
    multiplicity together with the caller's leading dims at their true extents;
    everything past those dims folds away, so a block bills what it was handed
    and positions no axis of its own.
    """
