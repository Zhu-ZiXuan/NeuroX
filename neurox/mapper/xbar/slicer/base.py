"""Slicer ABC and :class:`SlicingPlan`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SlicingPlan:
    """Per-slice per-digit decomposition of an integer value tensor.

    Attributes:
        values: Decomposed tensor, trailing-2 dims ``[slice_num, digit_num]``;
            each entry is one signed digit.
        slice_weights: Length-``slice_num`` int tensor of slice positional
            weights ``[1, R, R², ...]`` (LSB first), ``R`` the slice radix.
        digit_weights: Length-``digit_num`` int tensor of digit positional
            weights ``[1, r, r², ...]`` (LSB first), ``r`` the per-digit radix.
        value_range: Inclusive ``(lo, hi)`` integer range one input scalar
            can occupy under this slicer's strategy.
    """

    values: Tensor
    slice_weights: Tensor
    digit_weights: Tensor
    value_range: tuple[int, int]


class Slicer(ABC):
    """Abstract value-domain decomposer.

    Every runtime method receives the same keyword-only trio
    ``digit_count``, ``digit_radix``, ``digit_range``. :meth:`slice` returns
    a :class:`SlicingPlan` whose ``values`` has trailing-2
    ``[slice_num, digit_num]``.
    """

    @abstractmethod
    def value_range(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> tuple[int, int]:
        """Algorithm-side integer range this slicer can encode."""
        raise NotImplementedError

    @abstractmethod
    def slice_radix(
        self,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> int:
        """Per-slice positional radix; drives the downstream shift-add reduction."""
        raise NotImplementedError

    @abstractmethod
    def slice(
        self,
        x: Tensor,
        *,
        digit_count: int,
        digit_radix: int,
        digit_range: tuple[int, int],
    ) -> SlicingPlan:
        """Decompose ``x`` into ``[..., slice_num, digit_num]``."""
        raise NotImplementedError
