"""Slicer ABC for value-domain decomposition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Slicer(ABC):
    """Abstract value-domain decomposer.

    A slicer turns an integer-valued tensor into a digit tensor with
    trailing-2 axes ``[slice_num, digit_count]``. Concrete subclasses bind
    to a specific xbar-cell geometry through their own constructor; only
    the externally observable surface — ``value_range``, ``slice_radix``,
    and ``slice(x)`` — is declared here.
    """

    @property
    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer range this slicer can encode."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_radix(self) -> int:
        """Per-slice positional radix; drives the downstream shift-add reduction."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_weights(self) -> tuple[int, ...]:
        """LSB-first positional weight of each slice ``(1, R, R², …, R^(slice_num-1))``."""
        raise NotImplementedError

    @abstractmethod
    def slice(self, x: Tensor) -> Tensor:
        """Decompose ``x`` into trailing-2 ``[slice_num, digit_count]`` digit slots."""
        raise NotImplementedError
