"""Pluggable crossbar-cell abstraction.

See Also:
    docs/reference/primitive/xbar/cell/family.md
    docs/system_design/xbar_solve.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, DcopBase, ModuleBase, PolicyBase, SnapBase


class XbarCellConfig(ConfigBase, ABC):
    """Base class for crossbar-cell configurations."""

    def validate(self) -> None:
        """Validate parameter ranges; the base accepts every value."""


class XbarCellPolicy(PolicyBase, ABC):
    """Base class for crossbar-cell nonideality policies."""


class XbarCellSnap(SnapBase):
    """Base class for per-call cell snapshots."""


class XbarCellDcop(DcopBase):
    """Condensed branch working point of one cell DC evaluation.

    A model-specific convergence residual is no field here: it travels on the
    probe record of the model that computes it.
    """

    i__uA: Tensor
    """Branch current, positive bit-line into source-line. Shape: `[..., col, row]`."""
    di_dvbl__uS: Tensor
    """BL-side branch conductance ∂I/∂V_BL, non-negative. Shape: `[..., col, row]`."""
    di_dvsl__uS: Tensor
    """SL-side branch conductance ∂I/∂V_SL, non-positive. Shape: `[..., col, row]`."""


class XbarCell[ConfigT: XbarCellConfig, PolicyT: XbarCellPolicy, SnapT: XbarCellSnap, DcopT: XbarCellDcop](
    ModuleBase[ConfigT, PolicyT],
    ABC,
):
    """Condensed two-terminal crossbar-cell interface.

    An implementation must return the same branch triple from `solve_branch`
    and `solve_dc` for identical inputs and snap, and must read every per-call
    control from the snap rather than from `self`.

    The two conductance signs are an unchecked contract: nothing validates
    them, and a solve that assembles a Jacobian from the wrong sign corrupts
    its operating point silently instead of raising.

    Both solve methods run inside the compiled solver leaf, so an
    implementation stays traceable there — no in-place tensor write, and no
    Python branch on a tensor value.

    Args:
        inst_shape: Per-instance shape `(..., col, row)`.
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

    @abstractmethod
    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
    ) -> SnapT:
        """Sample one per-call snap of the cell's fabricated state.

        Args:
            control: Control-line drive [V] at each cell's own access-device
                gate.
                Shape: `[..., col, row]`.
            shape: Per-call broadcast shape `(..., col, row)` the owned device
                snaps fill their tensor fields at.
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
            w_state_idx: State-index tensor.
                Shape: `[*inst_shape]`.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_branch(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Solve the condensed branch current and terminal conductances.

        Args:
            v_bl__V: Bit-line node voltage.
                Shape: `[..., col, row]`.
            v_sl__V: Source-line node voltage.
                Shape: `[..., col, row]`.
            snap: Per-call snap from `snapshot`.

        Returns:
            `(i__uA, di_dvbl__uS, di_dvsl__uS)` — branch current positive
            BL → SL, ∂I/∂V_BL non-negative, and ∂I/∂V_SL non-positive, all
            three at one shape.
            Shape: `[..., col, row]`.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> DcopT:
        """Full branch DC working point, including internal-node state.

        Args:
            v_bl__V: Bit-line node voltage.
                Shape: `[..., col, row]`.
            v_sl__V: Source-line node voltage.
                Shape: `[..., col, row]`.
            snap: Per-call snap from `snapshot`.

        Returns:
            Concrete `XbarCellDcop` subclass with the branch working point and
            internal-node voltages.
        """
        raise NotImplementedError
