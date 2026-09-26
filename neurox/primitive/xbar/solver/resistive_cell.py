"""Resistive-cell role consumed by array DC solvers."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

__all__ = [
    "ResistiveCell",
    "ResistiveCellDcop",
]


class ResistiveCellDcop(Protocol):
    """Branch current and terminal derivatives required by an array solver."""

    @property
    def i__uA(self) -> Tensor:
        """Branch current, positive BL to SL."""
        ...

    @property
    def di_dvbl__uS(self) -> Tensor:
        """BL-side branch derivative, non-negative."""
        ...

    @property
    def di_dvsl__uS(self) -> Tensor:
        """SL-side branch derivative, non-positive."""
        ...


class ResistiveCell[SnapT, DcopT: ResistiveCellDcop](Protocol):
    """Condensed branch evaluated against a caller-supplied snapshot.

    The snapshot and cell evaluation must support compiled tensor execution.

    Implementers must return local derivatives evaluated at the requested
    operating point with the same broadcast layout as the returned signal. A
    solver may invoke this transfer repeatedly on one held snapshot. Keep calls
    free of new physical sampling, programming, or energy billing; numerical
    iterations are not additional physical accesses.
    """

    @property
    def inst_shape(self) -> tuple[int, ...]: ...

    def solve_dc(self, *, v_bl__V: Tensor, v_sl__V: Tensor, snap: SnapT) -> DcopT: ...
