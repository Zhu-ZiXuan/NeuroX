"""1T1R-family DC-solver framework.

Owns:
  * :class:`Solver1T1RConfig` — abstract config base for concrete 1T1R
    solver families (each carries its own iteration counts and any other
    runtime-fixed numerical knobs).
  * :class:`Solver1T1R` — abstract solver base with a registry that maps
    config types to concrete solver implementations. Each concrete solver
    registers itself via ``@Solver1T1R.register_key(SomeConfig)``.
  * :class:`Solver1T1RDCOP` / :class:`Solver1T1RResiduals` — shared DC
    operating-point and residual containers used by every 1T1R solver
    flavour, so the upstream :class:`CircuitCore1T1R` interface stays
    solver-agnostic.

Topology-specific (2T2R, differential, …) families that share little
with 1T1R should define their own ``Solver…`` base in their own module
rather than try to subclass this one.

See also:
    docs/dev/modules/xbar/_1t1r/solver.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from neurox.analog.clamp_driver import ClampDriver
from neurox.common.mixin import RegistryMixin, ValidateMixin
from neurox.device import RRAM, RRAMSnapshot
from neurox.device.nmos import NMOS, NMOSSnapshot

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
    """Per-element absolute KCL residuals from one 1T1R DC solve [μA] / [V].

    Populated only when ``solve_dc(compute_residuals=True)``; the hot path
    leaves the whole bundle as ``None`` so the residual algebra (an extra
    NMOS + RRAM evaluation plus two wire-KCL sweeps plus two clamp
    evaluations) is pruned away.

    Attributes:
        cell__uA: ``|I_NMOS - I_RRAM|`` per cell.
            Shape: ``[..., num_col, num_row]``.
        wire_bl__uA: BL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        wire_sl__uA: SL wire KCL residual per node.
            Shape: ``[..., num_col, num_row]``.
        clamp_bl__V: ``|TIA(I_BL_port) - V_BL_clamp|`` per column.
            Shape: ``[..., num_col]``.
        clamp_sl__V: ``|SL_driver(I_SL_port) - V_SL_drive|`` per column.
            Shape: ``[..., num_col]``.
    """

    cell__uA: Tensor
    wire_bl__uA: Tensor
    wire_sl__uA: Tensor
    clamp_bl__V: Tensor
    clamp_sl__V: Tensor


@dataclass(frozen=True)
class Solver1T1RDCOP:
    """Complete steady-state solution of one 1T1R DC solve.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: ``[..., num_col]``.
        i_sl_driver: SL driver current [uA]. Shape: ``[..., num_col]``.
        v_bl_node: BL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        v_sl_node: SL node voltages [V]. Shape: ``[..., num_col, num_row]``.
        v_x_node: Internal access-transistor drain voltages [V].
            Shape: ``[..., num_col, num_row]``.
        i_cell: Cell currents [uA]. Shape: ``[..., num_col, num_row]``.
        v_bl_clamp: BL clamp voltages [V]. Shape: ``[..., num_col]``.
        v_sl_drive: SL drive voltages [V]. Shape: ``[..., num_col]``.
        residuals: Optional per-element residual diagnostics. Hot path
            sets this to ``None`` — the extra KCL evaluations are skipped
            entirely. Calibration / debug paths call
            ``solve_dc(compute_residuals=True)`` to fill it.
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    v_x_node: Tensor
    i_cell: Tensor
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
    every concrete 1T1R solver needs exactly the same four boundary
    actors (RRAM, NMOS, BL driver, SL driver). Other topologies (2T2R,
    differential, …) define their own family base with their own
    boundary-actor signature.
    """

    @classmethod
    def from_config(
        cls,
        *,
        config: Solver1T1RConfig,
        rram: RRAM,
        nmos: NMOS,
        bl_driver: ClampDriver,
        sl_driver: ClampDriver,
    ) -> Solver1T1R:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            rram=rram,
            nmos=nmos,
            bl_driver=bl_driver,
            sl_driver=sl_driver,
        )

    @abstractmethod
    def solve_dc(
        self,
        *,
        v_wl_drive__V: Tensor,
        bl_segment_r__MOhm: Tensor,
        sl_segment_r__MOhm: Tensor,
        bl_segment_g__uS: Tensor,
        sl_segment_g__uS: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
        bl_driver_snapshot: object,
        sl_driver_snapshot: object,
        compute_residuals: bool = False,
    ) -> Solver1T1RDCOP:
        """Solve the fabricated 1T1R tile for one WL-drive tensor.

        Args:
            v_wl_drive__V: WL drive voltage tensor [V]. Shape: ``[..., 1, num_row]``.
            bl_segment_r__MOhm: 1-D BL segment resistances [MOhm]; index 0 is
                driver-to-first.
            sl_segment_r__MOhm: 1-D SL segment resistances [MOhm]; index 0 is
                driver-to-first.
            bl_segment_g__uS: BL segment conductances [uS], reciprocal of
                ``bl_segment_r__MOhm``.
            sl_segment_g__uS: SL segment conductances [uS], reciprocal of
                ``sl_segment_r__MOhm``.
            rram_snapshot: Per-solve RRAM snapshot.
            nmos_snapshot: Per-solve NMOS snapshot.
            bl_driver_snapshot: Per-solve BL driver snapshot.
            sl_driver_snapshot: Per-solve SL driver snapshot.
            compute_residuals: When True, populate
                :attr:`Solver1T1RDCOP.residuals` after convergence; when
                False (hot path) leaves it as ``None``.

        Returns:
            Complete steady-state solution for the current VMM.
        """
