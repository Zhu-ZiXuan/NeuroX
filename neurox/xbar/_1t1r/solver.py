"""1T1R-family DC-solver framework.

Hosts :class:`Solver1T1RConfig`, :class:`Solver1T1R`,
:class:`Solver1T1RDCOP`, and :class:`Solver1T1RResiduals` —
see ``docs/reference/xbar/_1t1r/solver.md`` for the design rationale
(per-family base, registry dispatch, residual container reuse).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from neurox.analog import Driver, DriverSnap
from neurox.analog.tia import TIA, TIASnap
from neurox.common.mixin import RegistryMixin, ValidateMixin
from neurox.xbar.cell import XbarCell

from .cell import XbarCell1T1RDCOP, XbarCell1T1RSnap

# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Solver1T1RConfig(ValidateMixin):
    """Abstract base for 1T1R DC-solver fixed-knob configs.

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
# Result containers (shared across all 1T1R solvers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Solver1T1RResiduals:
    """Per-element absolute wire / clamp KCL residuals from one 1T1R DC solve [μA] / [V].

    Solver-owned residuals only: the wire-ladder and clamp-boundary KCL
    mismatches. The per-cell internal-KCL residual lives on the cell DCOP
    (``Solver1T1RDCOP.cell.residuals``). Populated only when
    ``solve_dc(compute_residuals=True)``; the hot path leaves the whole
    bundle as ``None`` so the residual algebra (two wire-KCL sweeps plus
    two clamp evaluations) is pruned away.

    Attributes:
        wire_bl__uA: BL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        wire_sl__uA: SL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        clamp_bl__V: ``|TIA(I_BL_port) - V_BL_clamp|`` per column.
            Shape: ``[..., num_col]``.
        clamp_sl__V: ``|SL_driver(I_SL_port) - V_SL_drive|`` per column.
            Shape: ``[..., num_col]``.
    """

    wire_bl__uA: Tensor
    wire_sl__uA: Tensor
    clamp_bl__V: Tensor
    clamp_sl__V: Tensor


@dataclass(frozen=True)
class Solver1T1RDCOP:
    """Complete steady-state solution of one 1T1R DC solve.

    The condensed cell working point (branch current, signed terminal
    conductances, internal access-node voltage, and the per-cell KCL
    residual) is carried on :attr:`cell`; the solver owns only the wire
    and clamp boundary state.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: ``[..., num_col]``.
        i_sl_driver: SL driver current [uA]. Shape: ``[..., num_col]``.
        v_bl_node: BL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        v_sl_node: SL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        cell: Condensed cell DC working point at the converged node
            voltages, including the internal access-node voltage and the
            optional per-cell internal-KCL residual.
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
    cell: XbarCell1T1RDCOP
    v_bl_clamp: Tensor
    v_sl_drive: Tensor
    residuals: Solver1T1RResiduals | None


# ---------------------------------------------------------------------------
# Solver base + registry
# ---------------------------------------------------------------------------


class Solver1T1R(RegistryMixin[type["Solver1T1RConfig"], "Solver1T1R"], ABC):
    """Abstract base for 1T1R DC solvers with config-keyed dispatch.

    Each concrete solver registers itself against the :class:`Solver1T1RConfig`
    subclass it consumes via ``@Solver1T1R.register_key(SomeSolverConfig)``;
    callers reach it through :meth:`Solver1T1R.from_config`. Solvers are
    plain stateless tool classes — not ``nn.Module``, no buffers, no
    parameters — so the base intentionally stays minimal.

    The ``from_config`` signature is fully explicit for the 1T1R topology:
    every concrete 1T1R solver needs exactly the same three boundary
    actors (the pluggable cell, BL driver, SL driver). The cell owns the
    two-terminal device branch and condenses any internal node; the solver
    drives only the wire ladders and clamp boundaries. Other topologies
    (2T2R, differential, …) define their own family base with their own
    boundary-actor signature.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        config: Solver1T1RConfig,
        cell: XbarCell[XbarCell1T1RSnap, XbarCell1T1RDCOP],
        bl_driver: TIA,
        sl_driver: Driver,
    ) -> None:
        """Bind the solver to its boundary actors; concrete subclasses do the real init."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        *,
        config: Solver1T1RConfig,
        cell: XbarCell[XbarCell1T1RSnap, XbarCell1T1RDCOP],
        bl_driver: TIA,
        sl_driver: Driver,
    ) -> Solver1T1R:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            cell=cell,
            bl_driver=bl_driver,
            sl_driver=sl_driver,
        )

    @abstractmethod
    def solve_dc(
        self,
        *,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        cell_snap: XbarCell1T1RSnap,
        bl_driver_snap: TIASnap,
        sl_driver_snap: DriverSnap,
        compute_residuals: bool = False,
    ) -> Solver1T1RDCOP:
        """Solve the fabricated 1T1R tile for one cell snap.

        Args:
            bl_segment_r__MOhm: 1-D BL segment resistances [MOhm]; index 0 is
                driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances [MOhm]; index 0 is
                driver-to-first.
            bl_segment_g__uS: BL segment conductances [uS], reciprocal of
                ``bl_segment_r__MOhm``.
            sl_segment_g__uS: SL segment conductances [uS], reciprocal of
                ``sl_segment_r__MOhm``.
            cell_snap: Per-solve cell snap bundling the device
                snaps and the per-cell control-line (WL) drive.
            bl_driver_snap: Per-solve BL driver snap.
            sl_driver_snap: Per-solve SL driver snap.
            compute_residuals: When True, populate
                :attr:`Solver1T1RDCOP.residuals` after convergence; when
                False (hot path) leaves it as ``None``.

        Returns:
            Complete steady-state solution for the current VMM.
        """
