"""Shared base for digital, integer-exact circuit modules.

See Also:
    docs/internals/primitive/digital/base.md
"""

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
    """Base for digital, integer-exact circuit blocks."""

    def _sample_fabricate_mismatch(self) -> None:
        pass
