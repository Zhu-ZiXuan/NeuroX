"""Pluggable crossbar-cell abstraction.

See also:
    docs/internals/primitive/xbar/cell/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class XbarCellConfig(ConfigBase, ABC):
    """Base class for crossbar-cell configurations."""

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


class XbarCellPolicy(PolicyBase, ABC):
    """Base class for crossbar-cell nonideality policies."""


@dataclass(frozen=True)
class XbarCellSnap:
    """Base class for per-call cell snapshots."""


@dataclass(frozen=True)
class XbarCellDcop:
    """Condensed branch working point of one cell DC evaluation.

    Attributes:
        i__uA: Branch current, positive bit-line into source-line.
            Shape: ``[..., col, row]``.
        di_dvbl__uS: ``∂I/∂V_BL``, the BL-side branch conductance
            the wire Jacobian needs (non-negative).
            Shape: ``[..., col, row]``.
        di_dvsl__uS: ``∂I/∂V_SL``, the SL-side branch conductance
            (non-positive). Shape: ``[..., col, row]``.
    """

    i__uA: Tensor
    di_dvbl__uS: Tensor
    di_dvsl__uS: Tensor


SnapT = TypeVar("SnapT", bound=XbarCellSnap)
DCOPT = TypeVar("DCOPT", bound=XbarCellDcop)
ConfigT = TypeVar("ConfigT", bound=XbarCellConfig)
PolicyT = TypeVar("PolicyT", bound=XbarCellPolicy)


class XbarCell(
    ModuleBase[ConfigT, PolicyT],
    Generic[ConfigT, PolicyT, SnapT, DCOPT],
    ABC,
):
    """Condensed two-terminal crossbar-cell interface.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite per-device nonideality policy.
        inst_shape: Per-instance shape ``(*prefix, col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    is_profile_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        del dtype, T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @abstractmethod
    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
        t_elapsed: float,
    ) -> SnapT:
        """Sample one per-call snap of the cell's fabricated state.

        Args:
            control: Per-cell control-line drive [V] (the gate / select
                voltage of the cell's access device). Shape broadcasts to
                ``[..., col, row]``.
            shape: Per-call broadcast shape ``(*leading, col, row)`` the
                owned device snaps fill their tensor fields at.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view, forwarded to the device
                snaps; ``None`` returns the full view.
            t_elapsed: Time elapsed since programming [s], for any
                time-dependent device read state.

        Returns:
            Per-call snap bundling the device snaps and control.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w_state_idx: Tensor) -> None:
        """Program the cell's storage device from a state-index tensor.

        Args:
            w_state_idx: State-index tensor at ``self.inst_shape``.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: SnapT,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Solve the condensed branch current and terminal conductances.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            ``(i__uA, di_dvbl__uS, di_dvsl__uS)`` — branch current [uA]
            (positive BL → SL), ``∂I/∂V_BL`` [uS] (non-negative), and
            ``∂I/∂V_SL`` [uS] (non-positive). Each shape
            ``[..., col, row]``.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_dc(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: SnapT,
    ) -> DCOPT:
        """Full branch DC working point, including internal-node state.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Concrete :class:`XbarCellDcop` subclass with the branch
            working point and internal-node voltages.
        """
        raise NotImplementedError

    @abstractmethod
    def dynamic_energy(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        dcop: DCOPT,
        snap: SnapT,
    ) -> Tensor:
        """Per-cell capacitance switching energy [fJ].

        Excludes wire-segment capacitances.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            dcop: Converged DCOP from :meth:`solve_dc`, carrying any
                internal-node voltages the cap formulas need.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Per-cell switching energy [fJ]. Shape: ``[..., col, row]``.
        """
        raise NotImplementedError
