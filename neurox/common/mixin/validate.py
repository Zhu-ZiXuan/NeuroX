"""Runtime-validation mixin for frozen config dataclasses.

See also:
    docs/internals/config_and_construction.md
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise


class ValidateMixin:
    """Mixin providing runtime-check helpers for ``validate_*`` methods.

    Configs inherit this mixin and call the helpers through ``self``
    inside their own ``validate*`` methods. The mixin contributes only
    helper static methods — every config defines its own
    ``__post_init__`` (calling ``self.validate()``) and ``validate``.
    """

    @staticmethod
    def _require_pos(value: float, name: str) -> None:
        """Raise if ``value`` is not strictly positive."""
        if not (value > 0.0):
            raise ValueError(f"require: {name} ({value}) > 0")

    @staticmethod
    def _require_nonneg(value: float, name: str) -> None:
        """Raise if ``value`` is negative."""
        if not (value >= 0.0):
            raise ValueError(f"require: {name} ({value}) >= 0")

    @staticmethod
    def _require_strictly_increasing(seq: Iterable[int | float], name: str) -> None:
        """Raise if ``seq`` is not strictly increasing."""
        for prev, curr in pairwise(seq):
            if not (curr > prev):
                raise ValueError(f"require: {name} strictly increasing; got {prev} >= {curr}")

    @staticmethod
    def _require_strictly_decreasing(seq: Iterable[int | float], name: str) -> None:
        """Raise if ``seq`` is not strictly decreasing."""
        for prev, curr in pairwise(seq):
            if not (curr < prev):
                raise ValueError(f"require: {name} strictly decreasing; got {prev} <= {curr}")

    @staticmethod
    def _require_min_length(seq: Iterable[object], min_len: int, name: str) -> None:
        """Raise if ``len(seq) < min_len``."""
        n = sum(1 for _ in seq)
        if n < min_len:
            raise ValueError(f"require: len({name}) ({n}) >= {min_len}")
