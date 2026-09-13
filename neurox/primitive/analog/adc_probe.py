"""ADC input probe records."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from neurox.common.recorder import RecordBase, RecorderBase


class AdcRecord(RecordBase, ABC):
    # === For subclass to implement or override ===

    @abstractmethod
    def input_name(self) -> str:
        """Return the recorded input quantity's unit-bearing name."""
        raise NotImplementedError

    @abstractmethod
    def input_value(self) -> Tensor:
        """Return the scalar decision input at every conversion position."""
        raise NotImplementedError


class AdcProber(RecorderBase[AdcRecord]):
    """Capture ADC input records."""
