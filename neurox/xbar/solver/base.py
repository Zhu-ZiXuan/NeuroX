"""Topology-agnostic SL/BL IR-drop DC-solver framework.

Hosts :class:`SolverConfig`, :class:`Solver`, :class:`SolverDCOP`, and
:class:`SolverResiduals` — see ``docs/internals/xbar/solver.md`` for
the design rationale (registry dispatch, residual container reuse).

The solver drives only the two wire ladders and the two clamp boundaries.
The cell and both clamp drivers are per-call, method-generic parameters of
:meth:`Solver.solve_dc` rather than construction-time fields, so the
solver class is non-generic and stateless (it holds only its config). Any
SL/BL topology whose cell condenses to one two-terminal branch reuses the
same solver — the topology lives entirely in the supplied cell.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import Tensor

from neurox.common.mixin import RegistryMixin, ValidateMixin
from neurox.xbar.cell import XbarCell, XbarCellDCOP, XbarCellSnap

from .clamp import ClampDriver, ClampSnap

# ---------------------------------------------------------------------------
# Per-call method-generic type vars
# ---------------------------------------------------------------------------

# Bound only inside the solve-method signatures so mypy infers them per
# call and the solver class itself stays non-generic. ``CellSnapT`` /
# ``CellDCOPT`` carry the cell's own bounds (mirroring ``XbarCell``); the
# driver snaps are bound to ``ClampSnap``, matching ``ClampDriver`` — the
# snap carries the injected reference voltage the solver seeds from.
CellSnapT = TypeVar("CellSnapT", bound=XbarCellSnap)
CellDCOPT = TypeVar("CellDCOPT", bound=XbarCellDCOP)
BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)

# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig(ValidateMixin):
    """Abstract base for DC-solver fixed-knob configs.

    Empty by design — each concrete solver carries its own subclass with
    iteration counts and any other compile-time-constant numerical knobs.
    Pure algorithmic safety constants (Newton damping caps, Jacobian
    floors) are method-intrinsic and live as class attributes on the
    concrete solver class, NOT in this config tree.

    Solvers have **no Policy** — there is no per-source nonideality
    toggle in a pure numerical method; every knob is either a fixed
    design constant (here) or a method-intrinsic safety bound (on the
    concrete solver class).
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Hook for subclasses to enforce parameter ranges."""


# ---------------------------------------------------------------------------
# Result containers (shared across all solvers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverResiduals:
    """Per-element absolute wire / clamp KCL residuals from one DC solve [μA] / [V].

    Solver-owned residuals only: the wire-ladder and clamp-boundary KCL
    mismatches. The per-cell internal-KCL residual lives on the cell DCOP
    (``SolverDCOP.cell.residuals``). Populated only when
    ``solve_dc(compute_residuals=True)``; the hot path leaves the whole
    bundle as ``None`` so the residual algebra (two wire-KCL sweeps plus
    two clamp evaluations) is pruned away.

    Attributes:
        wire_bl__uA: BL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        wire_sl__uA: SL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        clamp_bl__V: ``|bl_driver(I_BL_port) - V_BL_clamp|`` per column.
            Shape: ``[..., num_col]``.
        clamp_sl__V: ``|sl_driver(I_SL_port) - V_SL_drive|`` per column.
            Shape: ``[..., num_col]``.
    """

    wire_bl__uA: Tensor
    wire_sl__uA: Tensor
    clamp_bl__V: Tensor
    clamp_sl__V: Tensor


@dataclass(frozen=True)
class SolverDCOP(Generic[CellDCOPT]):
    """Complete steady-state solution of one DC solve.

    The condensed cell working point (branch current, signed terminal
    conductances, internal node voltage, and the per-cell KCL residual) is
    carried on :attr:`cell`; the solver owns only the wire and clamp
    boundary state. ``CellDCOPT`` is the concrete cell DCOP type supplied
    to :meth:`Solver.solve_dc`, so the returned DCOP keeps the cell's exact
    working-point type end to end.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: ``[..., num_col]``.
        i_sl_driver: SL driver current [uA]. Shape: ``[..., num_col]``.
        v_bl_node: BL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        v_sl_node: SL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        cell: Condensed cell DC working point at the converged node
            voltages, including the internal node voltage and the optional
            per-cell internal-KCL residual.
        v_bl_clamp: BL clamp voltages [V]. Shape: ``[..., num_col]``.
        v_sl_drive: SL drive voltages [V]. Shape: ``[..., num_col]``.
        residuals: Optional per-element wire / clamp residual diagnostics.
            Hot path sets this to ``None`` — the extra KCL evaluations are
            skipped entirely. Calibration / debug paths call
            ``solve_dc(compute_residuals=True)`` to fill it.
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    cell: CellDCOPT
    v_bl_clamp: Tensor
    v_sl_drive: Tensor
    residuals: SolverResiduals | None


# ---------------------------------------------------------------------------
# Solver base + registry
# ---------------------------------------------------------------------------


class Solver(RegistryMixin[type["SolverConfig"], "Solver"], ABC):
    """Abstract base for SL/BL IR-drop DC solvers with config-keyed dispatch.

    Each concrete solver registers itself against the :class:`SolverConfig`
    subclass it consumes via ``@Solver.register_key(SomeSolverConfig)``;
    callers reach it through :meth:`Solver.from_config`. Solvers are plain
    stateless tool classes — not ``nn.Module``, no buffers, no parameters,
    and no boundary-actor fields — so the base intentionally stays minimal
    and ``from_config`` takes only the config.

    The cell and the two clamp drivers are per-call, method-generic
    parameters of :meth:`solve_dc` rather than construction-time bindings.
    The cell owns the two-terminal device branch and condenses any internal
    node; the solver drives only the wire ladders and clamp boundaries. Any
    SL/BL topology whose cell condenses to one branch reuses this solver
    unchanged — the topology lives entirely in the supplied cell.
    """

    @abstractmethod
    def __init__(self, *, config: SolverConfig, series_axis: int = -1) -> None:
        """Bind the solver to its config and layout axis; subclasses do the real init.

        Args:
            config: The concrete solver's fixed-knob config.
            series_axis: Construction-time index of the caller's series
                (wire-ladder) axis among the two trailing cell-grid axes.
                The core passes its layout's value; ``-1`` is canonical.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, *, config: SolverConfig, series_axis: int = -1) -> Solver:
        """Build the concrete impl registered for ``type(config)``.

        Args:
            config: Selects the impl (registry key) and its numerical knobs.
            series_axis: Layout series-axis index threaded to the impl's
                ``__init__`` (default ``-1``, canonical series-last).
        """
        impl = cls._lookup_impl(type(config))
        return impl(config=config, series_axis=series_axis)

    @abstractmethod
    def solve_dc(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell: XbarCell[CellSnapT, CellDCOPT],
        cell_snap: CellSnapT,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
        compute_residuals: bool = False,
    ) -> SolverDCOP[CellDCOPT]:
        """Solve the fabricated tile for one cell snap.

        Args:
            bl_segment_r__MOhm: 1-D BL segment resistances [MOhm]; index 0 is
                driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances [MOhm]; index 0 is
                driver-to-first.
            bl_segment_g__uS: BL segment conductances [uS], reciprocal of
                ``bl_segment_r__MOhm``.
            sl_segment_g__uS: SL segment conductances [uS], reciprocal of
                ``sl_segment_r__MOhm``.
            cell: Pluggable cell; owns the device branch and condenses any
                internal node.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                per-cell control-line (WL) drive.
            bl_driver: BL clamp driver.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver: SL clamp driver.
            sl_driver_snap: Per-solve SL driver snap.
            compute_residuals: When True, populate
                :attr:`SolverDCOP.residuals` after convergence; when False
                (hot path) leaves it as ``None``.

        Returns:
            Complete steady-state solution for the current VMM.
        """
