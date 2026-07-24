"""Topology-agnostic SL/BL IR-drop DC-solver framework.

Hosts :class:`SolverConfig`, :class:`Solver`, and :class:`SolverDcop`.

See also:
    docs/reference/primitive/xbar/solver/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor

from neurox.common import ConfigBase
from neurox.common.mixin import RegistryMixin
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from .clamp import ClampDriver, ClampSnap

# ---------------------------------------------------------------------------
# Per-call method-generic type vars
# ---------------------------------------------------------------------------

# Bound only inside the solve-method signatures so mypy infers them per
# call and the solver class itself stays non-generic.
CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDcop)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)

# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig(ConfigBase, ABC):
    """Abstract base for DC-solver fixed-knob configs.

    Each concrete solver carries its own subclass with iteration counts and
    any other compile-time-constant numerical knobs.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


# ---------------------------------------------------------------------------
# Result containers (shared across all solvers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverDcop(Generic[CellDCOPT]):
    """Complete steady-state solution of one DC solve.

    The condensed cell working point (branch current, signed terminal
    conductances, and internal node voltage) is carried on :attr:`cell`;
    the solver owns only the wire and clamp boundary state.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: ``[..., num_col]``.
        i_sl_driver: SL driver current [uA]. Shape: ``[..., num_col]``.
        v_bl_node: BL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        v_sl_node: SL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        cell: Condensed cell DC working point at the converged node
            voltages, including the internal node voltage.
        v_bl_clamp: BL clamp voltages [V]. Shape: ``[..., num_col]``.
        v_sl_drive: SL drive voltages [V]. Shape: ``[..., num_col]``.
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    cell: CellDCOPT
    v_bl_clamp: Tensor
    v_sl_drive: Tensor


# ---------------------------------------------------------------------------
# Solver base + registry
# ---------------------------------------------------------------------------


class Solver(RegistryMixin[type["SolverConfig"], "Solver"], ABC):
    """Abstract base for SL/BL IR-drop DC solvers with config-keyed dispatch.

    Each concrete solver registers itself against the :class:`SolverConfig`
    subclass it consumes via ``@Solver.register_key(SomeSolverConfig)``;
    callers reach it through :meth:`Solver.from_config`. Solvers are plain
    stateless tool classes (not ``nn.Module``); the cell and the two clamp
    drivers are per-call, method-generic parameters of :meth:`solve_dc`.
    """

    @abstractmethod
    def __init__(self, *, config: SolverConfig) -> None:
        """Bind the solver to its config.

        Args:
            config: The concrete solver's fixed-knob config.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, *, config: SolverConfig) -> Solver:
        """Build the concrete impl registered for ``type(config)``.

        Args:
            config: Selects the impl (registry key) and its numerical knobs.
        """
        impl = cls._lookup_impl(type(config))
        return impl(config=config)

    @abstractmethod
    def solve_dc(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[Any, Any, CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> SolverDcop[CellDCOPT]:
        """Solve the fabricated tile for one cell snap.

        Args:
            bl_segment_r__MOhm: 1-D BL segment resistances; index 0 is
                driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances; index 0 is
                driver-to-first.
            bl_segment_g__uS: BL segment conductances, reciprocal of
                ``bl_segment_r__MOhm``.
            sl_segment_g__uS: SL segment conductances, reciprocal of
                ``sl_segment_r__MOhm``.
            cell: Pluggable cell; owns the device branch and condenses any
                internal node.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-cell control-line (WL) drive.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.

        Returns:
            Complete steady-state solution for the current VMM.
        """
