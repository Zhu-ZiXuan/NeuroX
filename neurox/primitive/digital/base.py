"""Shared base for digital, integer-exact circuit modules.

See also:
    docs/internals/primitive/digital/base.md
"""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class DigitalConfig(ConfigBase, ABC):
    """Static PPA fields shared by every digital, integer-exact block.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class DigitalPolicy(PolicyBase):
    """Empty policy marker — integer-exact blocks carry no nonidealities."""


DigitalConfigT = TypeVar("DigitalConfigT", bound="DigitalConfig")


class DigitalBase(ModuleBase[DigitalConfigT, DigitalPolicy], Generic[DigitalConfigT], ABC):
    """Base for digital, integer-exact circuit blocks."""

    # === Circuit constant buffers ===

    _latency_per_op__ns: Tensor  # Shape: []

    def _register_latency_buffer(self, latency_per_op__ns: float) -> None:
        """Register the fixed per-operation latency source."""
        self.register_buffer(
            "_latency_per_op__ns",
            torch.tensor(latency_per_op__ns, dtype=torch.float32),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass
