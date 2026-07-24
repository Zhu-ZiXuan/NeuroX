"""Abstract crossbar pure-array primitive.

See also:
    docs/internals/primitive/xbar/array/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.xbar.solver import ClampDriver
from neurox.primitive.xbar.solver.clamp import ClampSnap

if TYPE_CHECKING:
    from ._1t1r import XbarArraySteadyState


class XbarArrayConfig(ConfigBase, ABC):
    """Static PPA fields common to crossbar-array configurations.

    Attributes:
        area_per_inst__um2: Cell-grid and wire area per instance [um²].
        leakage_per_inst__uW: Cell-grid and wire leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class XbarArrayPolicy(PolicyBase, ABC):
    """Base policy for crossbar-array nonidealities."""


ConfigT = TypeVar("ConfigT", bound=XbarArrayConfig)
PolicyT = TypeVar("PolicyT", bound=XbarArrayPolicy)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


class XbarArray(ModuleBase[ConfigT, PolicyT], ABC):
    """Base class for shape-independent crossbar arrays."""

    @property
    @abstractmethod
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the conductance grid ``(*inst, col, row)``."""
        raise NotImplementedError

    @abstractmethod
    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor whose shape matches the array's
                own ``(*inst, col_num, row_num)`` layout.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_array(
        self,
        v_wl: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT],
        bl_v_ref__V: Tensor,
        sl_driver: ClampDriver[SLSnapT],
        sl_v_ref__V: Tensor,
    ) -> XbarArraySteadyState:
        """Settle the array to DC under an analog WL drive.

        Args:
            v_wl: Analog WL drive [V]. Shape: ``[..., row_num]``.
            bl_driver: BL boundary clamp (structural ``ClampDriver`` role).
            bl_v_ref__V: BL-clamp reference tap, a 0-d scalar.
            sl_driver: SL boundary clamp (structural ``ClampDriver`` role).
            sl_v_ref__V: SL-drive reference tap, a 0-d scalar.

        Returns:
            :class:`XbarArraySteadyState` carrying the per-column BL port
            current [uA] and BL clamp voltage [V], both at full leading.
        """
        raise NotImplementedError

    def _sample_fabricate_mismatch(self) -> None:
        pass
