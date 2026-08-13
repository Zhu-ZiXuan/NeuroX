"""Runtime-check helpers for host `validate` methods."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise


class ValidateMixin:
    """Provide reusable predicates for `validate` methods.

    A host that inherits neither `ConfigBase` nor `PolicyBase` must invoke its
    validation explicitly.
    """

    @staticmethod
    def _require_gt(value: float, name: str, ref: float) -> None:
        """Require `value` > `ref`."""
        if not (value > ref):
            raise ValueError(f"require: {name} ({value}) > {ref}")

    @staticmethod
    def _require_ge(value: float, name: str, ref: float) -> None:
        """Require `value` >= `ref`."""
        if not (value >= ref):
            raise ValueError(f"require: {name} ({value}) >= {ref}")

    @staticmethod
    def _require_lt(value: float, name: str, ref: float) -> None:
        """Require `value` < `ref`."""
        if not (value < ref):
            raise ValueError(f"require: {name} ({value}) < {ref}")

    @staticmethod
    def _require_le(value: float, name: str, ref: float) -> None:
        """Require `value` <= `ref`."""
        if not (value <= ref):
            raise ValueError(f"require: {name} ({value}) <= {ref}")

    @staticmethod
    def _require_pos(value: float, name: str) -> None:
        """Require `value` > 0."""
        ValidateMixin._require_gt(value, name, 0.0)

    @staticmethod
    def _require_non_neg(value: float, name: str) -> None:
        """Require `value` >= 0."""
        ValidateMixin._require_ge(value, name, 0.0)

    @staticmethod
    def _require_increasing(seq: Iterable[int | float], name: str) -> None:
        """Require `seq` is strictly increasing."""
        for prev, curr in pairwise(seq):
            if not (curr > prev):
                raise ValueError(f"require: {name} strictly increasing; got {prev} >= {curr}")

    @staticmethod
    def _require_decreasing(seq: Iterable[int | float], name: str) -> None:
        """Require `seq` is strictly decreasing."""
        for prev, curr in pairwise(seq):
            if not (curr < prev):
                raise ValueError(f"require: {name} strictly decreasing; got {prev} <= {curr}")

    @staticmethod
    def _require_min_length(seq: Iterable[object], min_len: int, name: str) -> None:
        """Require `len(seq) >= min_len`."""
        n = sum(1 for _ in seq)
        if n < min_len:
            raise ValueError(f"require: len({name}) ({n}) >= {min_len}")
