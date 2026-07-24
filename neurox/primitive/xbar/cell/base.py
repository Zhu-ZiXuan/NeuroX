"""Pluggable crossbar-cell abstraction.

See also:
    docs/reference/primitive/xbar/cell/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase

# ---------------------------------------------------------------------------
# Config / policy / result bases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XbarCellConfig(ConfigBase, ABC):
    """Abstract base for crossbar-cell configs.

    Empty by design — each concrete cell carries its own subclass with
    the device configs, sizing, and per-cell capacitances it needs.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


@dataclass(frozen=True)
class XbarCellPolicy(PolicyBase, ABC):
    """Abstract marker base for crossbar-cell nonideality policies.

    Concrete cells carry a subclass bundling the per-device policies of
    their owned device children.
    """


@dataclass(frozen=True)
class XbarCellSnap:
    """Marker base for per-call snaps of a cell's fabricated state.

    Concrete cells carry a subclass bundling their device snaps plus
    any per-call control-line state the branch solve consumes.
    """


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


# ---------------------------------------------------------------------------
# Cell base + registry
# ---------------------------------------------------------------------------


SnapT = TypeVar("SnapT", bound=XbarCellSnap)
DCOPT = TypeVar("DCOPT", bound=XbarCellDcop)
ConfigT = TypeVar("ConfigT", bound=XbarCellConfig)
PolicyT = TypeVar("PolicyT", bound=XbarCellPolicy)


class XbarCell(
    ModuleBase[ConfigT, PolicyT],
    Generic[ConfigT, PolicyT, SnapT, DCOPT],
    ABC,
):
    """Solver-facing crossbar-cell contract.

    Defines the condensed two-terminal branch element the array solver
    consumes: :meth:`solve_branch` / :meth:`solve_dc` together with the
    :class:`XbarCellDcop` fields are the solver-cell ABI. A cell owns its
    device ``nn.Module`` children and condenses every internal node to
    expose that single branch.

    Construction and dispatch are not part of this contract: the
    config-keyed registry and ``from_config`` live on the family roots
    (e.g. :class:`XbarCell1t1r`), so a cell is built family-bounded by
    type rather than through a universal factory plus runtime narrowing.

    A cell is a non-reporting :class:`ModuleBase` leaf
    (``is_profile_target`` is ``False``): it self-accounts no static PPA.
    Its device children's physical area / leakage roll up through the
    owning core's PPA budget.
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
        """Register the instance with :class:`nn.Module`; subclass builds devices.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite per-device nonideality policy.
            inst_shape: Per-instance fabrication shape ``(*prefix, col, row)``.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.
        """
        del dtype, T__K  # consumed by the subclass init
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    def _sample_fabricate_mismatch(self) -> None:
        pass  # container: device mismatch is sampled through the cascade

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
            w_state_idx: State-index tensor at ``self._inst_shape``.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: SnapT,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condensed branch solve — lean, compile-safe hot path.

        Solves any internal cell node and returns only what the array
        wire Newton needs: the branch current and its two signed
        terminal conductances.

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

        Diagnostic / energy-side superset of :meth:`solve_branch`: returns
        the condensed branch quantities plus the cell's internal node
        voltages (carried on the concrete DCOP subclass). A cell that owns
        a per-cell KCL residual computes it at the converged internal node
        and emits it on its own probe channel.

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

        Sums the cell's per-node capacitance charge/discharge energy
        from the node voltages of the converged operating point.
        Excludes wire-segment capacitances (owned by the core) — only
        the per-cell node loading.

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
