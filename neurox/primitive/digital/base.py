"""Shared base for digital, integer-exact circuit modules.

See also:
    docs/internals/primitive/digital/base.md
"""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class DigitalConfig(ConfigBase, ABC):
    """Static PPA fields shared by every digital, integer-exact block.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DigitalPolicy(PolicyBase):
    """Empty policy marker — integer-exact blocks carry no nonidealities."""


DigitalConfigT = TypeVar("DigitalConfigT", bound="DigitalConfig")


class DigitalBase(ModuleBase[DigitalConfigT, DigitalPolicy], Generic[DigitalConfigT], ABC):
    """Base for digital, integer-exact circuit blocks."""

    def _sample_fabricate_mismatch(self) -> None:
        pass
