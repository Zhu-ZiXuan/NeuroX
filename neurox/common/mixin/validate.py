"""Runtime-check helpers for host ``validate`` methods.

See also:
    docs/internals/common/mixin/validate.md
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise


class ValidateMixin:
    """Grant a host a small set of runtime-check helpers.

    A host inherits this and calls the helpers through ``self`` from inside its
    own ``validate_*`` methods to assert runtime constraints that static typing
    cannot express — numeric bounds, monotonicity, length, cross-field
    relations. It is a pure helper: it owns no field, no construction hook, and
    no ``validate`` of its own, so inheriting it supplies the checks but does
    not by itself make a host validated — running validation at construction is
    the host's to wire.

    Host requirements:
        - Inherit ``ValidateMixin`` and reach the helpers through ``self``.
        - Declare a ``__post_init__`` that calls ``self.validate()`` so checks
          run at construction; the mixin installs no such hook itself.
        - Declare one or more ``validate_<group>()`` methods holding the checks
          (a single-field host may inline its check directly in ``validate()``).
        - Begin any ``validate()`` that overrides a base carrying its own checks
          with a leading ``super().validate()`` so the chain runs end to end.
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
