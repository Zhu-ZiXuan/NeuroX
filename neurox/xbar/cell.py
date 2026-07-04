"""Pluggable crossbar-cell abstraction.

A cell encapsulates the analog two-terminal device branch seen by the
array solver between a bit-line node and a source-line node: its DC
current, the signed branch conductances the wire Jacobian needs, and the
per-cell device-capacitance switching energy. Concrete cells own their
``nn.Module`` device children and expose a single condensed branch — any
internal device topology node is solved inside the cell, never by the
array solver.

See also:
    docs/reference/xbar/cell.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, RegistryMixin, ValidateMixin

# ---------------------------------------------------------------------------
# Config / policy / result bases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XbarCellConfig(ValidateMixin):
    """Abstract base for crossbar-cell configs.

    Empty by design — each concrete cell carries its own subclass with
    the device configs, sizing, and per-cell parasitic-cap densities it
    needs. A cell config has no PPA fields: the cell owns ``nn.Module``
    device children whose physical area / leakage roll up through the
    owning core's PPA budget, not through the cell.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


@dataclass(frozen=True)
class XbarCellPolicy:
    """Abstract marker base for crossbar-cell nonideality policies.

    Concrete cells carry a subclass bundling the per-device policies of
    their owned device children.
    """


@dataclass(frozen=True)
class XbarCellResiduals:
    """Marker base for per-cell DC-solve residual diagnostics.

    Concrete cells carry a subclass holding the absolute internal-KCL
    residual of their own condensation (for example an access-node
    current mismatch).
    """


@dataclass(frozen=True)
class XbarCellSnap:
    """Marker base for per-call snaps of a cell's fabricated state.

    Concrete cells carry a subclass bundling their device snaps plus
    any per-call control-line state the branch solve consumes.
    """


ResidualsT = TypeVar("ResidualsT", bound=XbarCellResiduals)


@dataclass(frozen=True)
class XbarCellDCOP(Generic[ResidualsT]):
    """Condensed branch working point of one cell DC evaluation.

    Attributes:
        i__uA: Branch current, positive bit-line into source-line.
            Shape: ``[..., col, row]``.
        di_dvbl__uS: ``∂I/∂V_BL``, the BL-side branch conductance
            the wire Jacobian needs (non-negative).
            Shape: ``[..., col, row]``.
        di_dvsl__uS: ``∂I/∂V_SL``, the SL-side branch conductance
            (non-positive). Shape: ``[..., col, row]``.
        residuals: Optional internal-KCL residual diagnostics; ``None``
            on the hot path. Populated by ``solve_dc(compute_residuals=True)``.
    """

    i__uA: Tensor
    di_dvbl__uS: Tensor
    di_dvsl__uS: Tensor
    residuals: ResidualsT | None


# ---------------------------------------------------------------------------
# Cell base + registry
# ---------------------------------------------------------------------------


SnapT = TypeVar("SnapT", bound=XbarCellSnap)
DCOPT = TypeVar("DCOPT", bound=XbarCellDCOP)


class XbarCell(
    FabricateMixin,
    nn.Module,
    RegistryMixin[type["XbarCellConfig"], "XbarCell"],
    Generic[SnapT, DCOPT],
    ABC,
):
    """Abstract base for pluggable crossbar cells with config-keyed dispatch.

    Parameterised by the concrete snap and DCOP types
    (``SnapT`` / ``DCOPT``) so each implementation declares those
    dataclasses once and the snap-consuming methods (:meth:`snapshot`,
    :meth:`solve_branch`, :meth:`solve_dc`, :meth:`dynamic_energy`) carry
    the concrete types without an LSP-narrowing override. The registry-impl
    slot is unparameterised because Python generics are invariant — each
    concrete impl binds the two type vars to its own subclasses.

    A cell owns its device ``nn.Module`` children (``FabricateMixin`` +
    ``nn.Module`` so manufacturing variation cascades and buffers move
    with ``.to`` / ``.eval``) and exposes one condensed two-terminal
    branch to the array solver. Each concrete cell registers itself
    against the :class:`XbarCellConfig` subclass it consumes via
    ``@XbarCell.register_key(SomeCellConfig)``; callers reach it through
    :meth:`from_config`.

    A cell carries **no PPA** — it is not a ``CircuitBase``. Its device
    children's physical area / leakage roll up through the owning core's
    PPA budget; the cell only contributes the per-VMM dynamic switching
    energy of its device capacitances via :meth:`dynamic_energy`.
    """

    def __init__(
        self,
        *,
        config: XbarCellConfig,
        policy: XbarCellPolicy,
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
        del policy, dtype, T__K  # consumed by the subclass init
        super().__init__()
        self.config = config
        self._inst_shape = inst_shape

    @classmethod
    def from_config(
        cls,
        *,
        config: XbarCellConfig,
        policy: XbarCellPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> XbarCell:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

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
        compute_residuals: bool = False,
    ) -> DCOPT:
        """Full branch DC working point, including internal-node state.

        Diagnostic / energy-side superset of :meth:`solve_branch`: returns
        the condensed branch quantities plus the cell's internal node
        voltages (carried on the concrete DCOP subclass) and, optionally,
        the internal-KCL residual.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            snap: Per-call snap from :meth:`snapshot`.
            compute_residuals: When True, populate
                :attr:`XbarCellDCOP.residuals`; when False (hot path)
                leaves it as ``None``.

        Returns:
            Concrete :class:`XbarCellDCOP` subclass with the branch
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
        """Per-cell device-capacitance switching energy [fJ].

        Sums the cell's internal device-capacitance charge/discharge
        energy from the node voltages of the converged operating point.
        Excludes wire-segment and control-line capacitances (owned by the
        core) — only the capacitances internal to the cell's devices.

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
