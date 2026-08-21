"""Resistive-cell role consumed by DC solvers."""

from typing import Protocol

from torch import Tensor

from neurox.common import SnapBase


class ResistiveDcop(Protocol):
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


class ResistiveCell[SnapT: SnapBase, DcopT: ResistiveDcop](Protocol):
    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> DcopT: ...
