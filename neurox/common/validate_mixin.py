"""Runtime-check helpers for host `validate` methods."""

from __future__ import annotations

from collections.abc import Iterable, Sized
from itertools import pairwise
from math import isfinite


class ValidateMixin:
    """Provide reusable predicates for host-defined `validate` methods.

    Every numeric predicate rejects non-finite operands.
    """

    # === Tools for subclass and internal use ===

    @staticmethod
    def _require_gt(value: float, name: str, ref: float) -> None:
        """Require finite `value` > finite `ref`."""
        if not (isfinite(value) and isfinite(ref) and value > ref):
            raise ValueError(f"require: {name} ({value}) > {ref}; both operands must be finite")

    @staticmethod
    def _require_ge(value: float, name: str, ref: float) -> None:
        """Require finite `value` >= finite `ref`."""
        if not (isfinite(value) and isfinite(ref) and value >= ref):
            raise ValueError(f"require: {name} ({value}) >= {ref}; both operands must be finite")

    @staticmethod
    def _require_lt(value: float, name: str, ref: float) -> None:
        """Require finite `value` < finite `ref`."""
        if not (isfinite(value) and isfinite(ref) and value < ref):
            raise ValueError(f"require: {name} ({value}) < {ref}; both operands must be finite")

    @staticmethod
    def _require_le(value: float, name: str, ref: float) -> None:
        """Require finite `value` <= finite `ref`."""
        if not (isfinite(value) and isfinite(ref) and value <= ref):
            raise ValueError(f"require: {name} ({value}) <= {ref}; both operands must be finite")

    @staticmethod
    def _require_pos(value: float, name: str) -> None:
        """Require finite `value` > 0."""
        ValidateMixin._require_gt(value, name, 0.0)

    @staticmethod
    def _require_non_neg(value: float, name: str) -> None:
        """Require finite `value` >= 0."""
        ValidateMixin._require_ge(value, name, 0.0)

    @staticmethod
    def _require_finite(value: float, name: str) -> None:
        """Require finite `value`."""
        if not isfinite(value):
            raise ValueError(f"require: {name} ({value}) finite")

    @staticmethod
    def _require_in_closed_interval(value: float, name: str, lower: float, upper: float) -> None:
        """Require finite `value` in the finite closed interval `[lower, upper]`."""
        if not (isfinite(value) and isfinite(lower) and isfinite(upper) and lower <= value <= upper):
            raise ValueError(f"require: {name} ({value}) in closed interval [{lower}, {upper}]")

    @staticmethod
    def _require_increasing(seq: Iterable[int | float], name: str) -> None:
        """Require `seq` contains finite values in strictly increasing order."""
        values = tuple(seq)
        for index, value in enumerate(values):
            ValidateMixin._require_finite(value, f"{name}[{index}]")
        for prev, curr in pairwise(values):
            if not (curr > prev):
                raise ValueError(f"require: {name} strictly increasing; got {prev} >= {curr}")

    @staticmethod
    def _require_decreasing(seq: Iterable[int | float], name: str) -> None:
        """Require `seq` contains finite values in strictly decreasing order."""
        values = tuple(seq)
        for index, value in enumerate(values):
            ValidateMixin._require_finite(value, f"{name}[{index}]")
        for prev, curr in pairwise(values):
            if not (curr < prev):
                raise ValueError(f"require: {name} strictly decreasing; got {prev} <= {curr}")

    @staticmethod
    def _require_non_empty(seq: Sized, name: str) -> None:
        """Require `seq` contains at least one item."""
        ValidateMixin._require_min_len(seq, name, 1)

    @staticmethod
    def _require_len(seq: Sized, name: str, expected: int) -> None:
        """Require `len(seq) == expected`."""
        actual = len(seq)
        if actual != expected:
            raise ValueError(f"require: len({name}) ({actual}) == {expected}")

    @staticmethod
    def _require_same_len(left: Sized, left_name: str, right: Sized, right_name: str) -> None:
        """Require two values have equal lengths."""
        left_len = len(left)
        right_len = len(right)
        if left_len != right_len:
            raise ValueError(f"require: len({left_name}) ({left_len}) == len({right_name}) ({right_len})")

    @staticmethod
    def _require_min_len(seq: Sized, name: str, min_len: int) -> None:
        """Require `len(seq) >= min_len`."""
        actual = len(seq)
        if actual < min_len:
            raise ValueError(f"require: len({name}) ({actual}) >= {min_len}")
