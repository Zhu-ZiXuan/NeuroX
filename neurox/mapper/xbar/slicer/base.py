"""Slicer ABC + :class:`SlicingResult`.

A :class:`Slicer` decomposes one integer scalar into per-slice +
per-digit components.  It is the *value-domain* operation in the
mapping pipeline; matrix tiling (`Tiler`) is a separate concern.

Output convention: ``slice(x, ...)`` always produces a tensor with
trailing-2 ``[slice_num, digit_num]`` on top of whatever shape
``x`` carries — so the same downstream plumbing handles both the
activation path (``digit_num`` is structural ``1`` for serial
decompositions) and the weight path (``digit_num =
w_digit_count``).

Runtime contract (§4 of ``temp/mapping.md``)
--------------------------------------------
Every runtime method takes the **same** three keyword-only
arguments — no ``**kwargs``, no ``value_range`` *input* (that
quantity is the slicer's *output* semantics, derived from the
slicer's static strategy plus the three runtime kwargs):

* ``digit_count``  — digit slots per slice (= ``digit_num`` in
  the output's trailing-2 shape).
* ``digit_radix``  — radix of one digit position.
* ``digit_range``  — integer range one physical digit supports.

A concrete slicer ignores parameters it does not consume but must
not change the signature.  The mapper translates xbar capabilities
into this fixed 3-kwarg shape; the slicer never sees raw xbar caps
and never receives a caller-supplied ``value_range``.

Strategy parameters (``slice_num``, encoding choices) live on the
instance.  Slicers are stateless apart from their strategy and may
be reused across xbars with different capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SlicingResult:
    """Per-slice per-digit decomposition of an integer value tensor.

    Attributes:
        values: Decomposed tensor.  Trailing-2 dims are
            ``[slice_num, digit_num]``; preceding dims match the
            input.  Each entry is one signed digit.
        slice_weights: Length-``slice_num`` int tensor of slice
            positional weights ``[1, R, R^2, ...]`` (LSB first),
            where ``R`` is the slice radix.  Used downstream by
            the macro's shift-add reduction.
        digit_weights: Length-``digit_num`` int tensor of digit
            positional weights ``[1, r, r^2, ...]`` (LSB first),
            where ``r`` is the per-digit radix.  ``digit_num = 1``
            mappers (serial activation) report ``[1]``.
        value_range: Inclusive ``(lo, hi)`` of the algorithm-side
            integer range one input scalar can occupy under this
            slicer's static strategy.  Computed by the slicer from
            its static parameters + the runtime digit capability —
            never supplied as input.
    """

    values: Tensor
    slice_weights: Tensor
    digit_weights: Tensor
    value_range: tuple[int, int]


class Slicer(ABC):
    """Abstract value-domain decomposer with a fixed 3-kwarg runtime contract.

    Every runtime method receives the same keyword-only trio:
    ``digit_count``, ``digit_radix``, ``digit_range``.  Concrete
    subclasses ignore parameters they do not consume; no subclass
    is permitted to widen or shrink this signature.  ``value_range``
    is the slicer's *output* (carried on :class:`SlicingResult` and
    queried via :meth:`value_range`), never an input.

    The unified output contract for :meth:`slice` is a
    :class:`SlicingResult` whose ``values`` tensor has trailing-2
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
        """Algorithm-side integer range this slicer can encode.

        Derived from the slicer's static strategy plus the runtime
        digit capability.  Result MUST agree with the reachable
        domain of :meth:`slice` under the same kwargs: any value
        outside the returned range is also unencodable, and any
        value inside it is encodable without error.
        """
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
    ) -> SlicingResult:
        """Decompose ``x`` into ``[..., slice_num, digit_num]``."""
        raise NotImplementedError
