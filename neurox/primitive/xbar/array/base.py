"""Abstract crossbar pure-array primitive.

See also:
    docs/reference/primitive/xbar/array/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.xbar.solver import ClampDriver
from neurox.primitive.xbar.solver.clamp import ClampSnap

if TYPE_CHECKING:
    from ._1t1r import XbarArraySteadyState


@dataclass(frozen=True)
class XbarArrayConfig(ConfigBase, ABC):
    """Static PPA fields common to every crossbar pure-array config.

    Attributes:
        area_per_inst__um2: Array (cell grid + wire infrastructure)
            silicon area per fabricated tile instance. Excludes the
            boundary drivers / DAC / reference, which are peers of the
            array and roll up separately.
        leakage_per_inst__uW: Array static leakage per fabricated tile
            instance. Same scope as ``area_per_inst__um2``.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
class XbarArrayPolicy(PolicyBase, ABC):
    """Abstract marker base for crossbar pure-array nonideality policies.

    Concrete arrays carry a subclass bundling the per-cell policy plus any
    array-level runtime knobs.
    """


ConfigT = TypeVar("ConfigT", bound=XbarArrayConfig)
PolicyT = TypeVar("PolicyT", bound=XbarArrayPolicy)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


class XbarArray(ModuleBase[ConfigT, PolicyT], ABC):
    """Abstract base for a shape-independent crossbar pure array.

    A pure array holds the cell grid, the wire parasitics, and the DC
    solver. The boundary drivers (WL DAC, BL clamp, SL drive) and the
    boundary voltage reference are peers of the array under the scheme
    macro — they are passed into :meth:`solve_array` per call, not owned
    here. The array settles to DC under an analog WL drive and returns the
    per-column BL port current plus BL clamp voltage the macro readout
    consumes.

    A reporting node (``is_profile_target`` default): a concrete array sets
    the bare per-instance PPA data (cell grid + wire infrastructure area /
    leakage, from config) that ``ProfileMixin`` aggregates into the reported
    ``area__um2`` / ``leakage__uW``.
    """

    @property
    @abstractmethod
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the conductance grid ``(*inst, phys_col, row)``."""
        raise NotImplementedError

    @abstractmethod
    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor whose shape matches the array's
                own ``(*prefix, phys_col_num, row_num)`` layout.
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
        pass  # container: cell mismatch is sampled through the cascade
