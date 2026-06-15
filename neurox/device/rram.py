"""RRAM device model with programming, read noise, and nonlinear I-V.

See also:
    docs/reference/device/rram.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ValidateMixin
from neurox.common.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)


@dataclass(frozen=True)
class RRAMConfig(ValidateMixin):
    """Static RRAM device configuration.

    Attributes:
        g_min__uS: Minimum programmable conductance [uS].
        nonlinearity_alpha: Hyperbolic-sine I-V nonlinearity factor [1/V].
        drift_decay_rate: Power-law drift exponent.
        drift_t0: Reference drift time [s].
        c_top__fF: Top-electrode parasitic capacitance per cell [fF].
        c_bot__fF: Bottom-electrode parasitic capacitance per cell [fF].
        read_thermal__uS: Gaussian read-noise sigma [uS].
        prog_gamma: Programming-variation model parameters.
        read_telegraph: Telegraph-noise model parameters.
        stuck_at: Stuck-at fault model parameters.
    """

    # --- Working range ---
    g_min__uS: float

    # --- I-V nonlinearity ---
    nonlinearity_alpha: float

    # --- Drift ---
    drift_decay_rate: float
    drift_t0: float

    # --- Per-cell parasitics ---
    c_top__fF: float
    c_bot__fF: float

    # --- Read thermal noise ---
    read_thermal__uS: float

    # --- Programming Gamma ---
    prog_gamma: StateDependentGammaConfig

    # --- Read telegraph noise ---
    read_telegraph: TelegraphConfig

    # --- Stuck-at fault ---
    stuck_at: StuckAtFaultConfig

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_range()
        self.validate_iv()
        self.validate_drift()
        self.validate_parasitics()
        self.validate_noise()

    def validate_range(self) -> None:
        self._require_nonneg(self.g_min__uS, "g_min__uS")

    def validate_iv(self) -> None:
        self._require_nonneg(self.nonlinearity_alpha, "nonlinearity_alpha")

    def validate_drift(self) -> None:
        self._require_nonneg(self.drift_decay_rate, "drift_decay_rate")
        self._require_nonneg(self.drift_t0, "drift_t0")

    def validate_parasitics(self) -> None:
        self._require_nonneg(self.c_top__fF, "c_top__fF")
        self._require_nonneg(self.c_bot__fF, "c_bot__fF")

    def validate_noise(self) -> None:
        # Nested *Config self-validates in its own __post_init__.
        self._require_nonneg(self.read_thermal__uS, "read_thermal__uS")


@dataclass(frozen=True)
class RRAMPolicy:
    """Per-source toggles selecting which RRAM nonidealities are active.

    Attributes:
        prog_gamma: Apply state-dependent programming Gamma at program time.
        stuck_at: Apply stuck-at faults at program time.
        read_telegraph: Apply telegraph noise at snapshot time.
        read_thermal: Apply Gaussian read noise at snapshot time.
    """

    prog_gamma: bool
    stuck_at: bool
    read_telegraph: bool
    read_thermal: bool


@dataclass(frozen=True)
class RRAMDCOP:
    """Device current and local differential conductance.

    Attributes:
        i__uA: Device current [uA].
        di_dv__uS: Local differential conductance [uS].
    """

    i__uA: Tensor
    di_dv__uS: Tensor


@dataclass(frozen=True)
class RRAMSnapshot:
    """Per-call read conductance snapshot.

    Attributes:
        g__uS: Sampled per-cell conductance [uS].
    """

    g__uS: Tensor


class RRAM(FabricateMixin, nn.Module):
    """Stateful RRAM array model."""

    g__uS: Tensor

    def __init__(
        self,
        *,
        config: RRAMConfig,
        policy: RRAMPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        g_max__uS: float,
    ) -> None:
        """Construct one stateful RRAM model.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality enable flags.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
            g_max__uS: Maximum programmable conductance [uS].
        """
        super().__init__()

        if not (g_max__uS > config.g_min__uS):
            raise ValueError(f"require: g_max__uS ({g_max__uS}) > config.g_min__uS ({config.g_min__uS})")

        self.config = config
        self.policy = policy
        self._inst_shape = inst_shape
        self.dtype = dtype
        self.T__K = T__K
        self.g_min__uS = config.g_min__uS
        self.g_max__uS = g_max__uS

        self.register_buffer("g__uS", torch.zeros((), dtype=dtype), persistent=False)

    @property
    def c_top__fF(self) -> float:
        """Top-electrode (BL-side) parasitic capacitance per cell [fF]."""
        return self.config.c_top__fF

    @property
    def c_bot__fF(self) -> float:
        """Bottom-electrode (Node-X-side) parasitic capacitance per cell [fF]."""
        return self.config.c_bot__fF

    def program(self, target_g__uS: Tensor, t_elapsed: float) -> None:
        """Program the stored conductance.

        Args:
            target_g__uS: Target conductance tensor [uS].
            t_elapsed: Time elapsed since programming [s].
        """
        g__uS = target_g__uS.to(dtype=self.dtype).clamp(self.g_min__uS, self.g_max__uS)
        g__uS = apply_state_dependent_gamma(g__uS, self.config.prog_gamma, enabled=self.policy.prog_gamma)
        if self.config.drift_decay_rate > 0.0 and t_elapsed > self.config.drift_t0:
            drift_factor = (t_elapsed / self.config.drift_t0) ** (-self.config.drift_decay_rate)
            g__uS = g__uS * drift_factor

        g__uS = apply_stuck_at_fault(
            x=g__uS,
            config=self.config.stuck_at,
            min_val=self.g_min__uS,
            max_val=self.g_max__uS,
            enabled=self.policy.stuck_at,
        )

        g__uS = g__uS.clamp(self.g_min__uS, self.g_max__uS)

        self.g__uS = g__uS

    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> RRAMSnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snapshot fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        g_view = self.g__uS.expand(shape) if shape else self.g__uS
        g = g_view if multi_coords is None else g_view[multi_coords]
        g = apply_telegraph_noise(g, self.config.read_telegraph, enabled=self.policy.read_telegraph)
        g = apply_gaussian(g, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
        g = g.clamp(self.g_min__uS, self.g_max__uS)
        return RRAMSnapshot(g__uS=g)

    def solve_dc(self, v__V: Tensor, snapshot: RRAMSnapshot) -> RRAMDCOP:
        """Evaluate current and differential conductance.

        Args:
            v__V: Device voltage [V]. Shape: arbitrary.
            snapshot: Conductance snapshot from :meth:`snapshot`.

        Returns:
            Current and local differential conductance at the requested
            voltage.
        """
        g__uS = snapshot.g__uS
        alpha = self.config.nonlinearity_alpha
        if alpha == 0.0:
            i__uA = g__uS * v__V
            di_dv__uS = g__uS.expand_as(i__uA)
        else:
            ax = alpha * v__V
            i__uA = g__uS * torch.sinh(ax) / alpha
            di_dv__uS = g__uS * torch.cosh(ax)
        return RRAMDCOP(i__uA=i__uA, di_dv__uS=di_dv__uS)
