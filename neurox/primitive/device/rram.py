"""Programmable-conductance RRAM device model.

See also:
    docs/reference/primitive/device/rram.md
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)


@dataclass(frozen=True)
class RramConfig(ConfigBase):
    """Static RRAM device configuration.

    Attributes:
        g_min__uS: Minimum programmable conductance.
        nonlinearity_alpha: Hyperbolic-sine I-V nonlinearity factor [1/V].
        drift_decay_rate: Power-law drift exponent.
        drift_t0: Reference drift time [s].
        c_top__fF: Top-electrode parasitic capacitance per cell.
        c_bot__fF: Bottom-electrode parasitic capacitance per cell.
        read_thermal__uS: Gaussian read-noise σ.
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
        self._require_non_neg(self.g_min__uS, "g_min__uS")

    def validate_iv(self) -> None:
        self._require_non_neg(self.nonlinearity_alpha, "nonlinearity_alpha")

    def validate_drift(self) -> None:
        self._require_non_neg(self.drift_decay_rate, "drift_decay_rate")
        self._require_non_neg(self.drift_t0, "drift_t0")

    def validate_parasitics(self) -> None:
        self._require_non_neg(self.c_top__fF, "c_top__fF")
        self._require_non_neg(self.c_bot__fF, "c_bot__fF")

    def validate_noise(self) -> None:
        # Nested *Config self-validates in its own __post_init__.
        self._require_non_neg(self.read_thermal__uS, "read_thermal__uS")


@dataclass(frozen=True)
class RramPolicy(PolicyBase):
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
class RramDcop:
    """Device current and local differential conductance.

    Attributes:
        i__uA: Device current.
        di_dv__uS: Local differential conductance.
    """

    i__uA: Tensor
    di_dv__uS: Tensor


@dataclass(frozen=True)
class RramSnap:
    """Per-call read conductance snap.

    Attributes:
        g__uS: Sampled per-cell conductance.
    """

    g__uS: Tensor


class Rram(ModuleBase[RramConfig, RramPolicy]):
    """Stateful RRAM array model."""

    # non-reporter: silicon rolls up to the owner
    is_profile_target: ClassVar[bool] = False

    g__uS: Tensor

    def __init__(
        self,
        *,
        config: RramConfig,
        policy: RramPolicy,
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
            T__K: Operating temperature.
            g_max__uS: Maximum programmable conductance.
        """
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if not (g_max__uS > config.g_min__uS):
            raise ValueError(f"require: g_max__uS ({g_max__uS}) > config.g_min__uS ({config.g_min__uS})")

        self.dtype = dtype
        self.T__K = T__K
        self.g_min__uS = config.g_min__uS
        self.g_max__uS = g_max__uS

        self.register_buffer("g__uS", torch.zeros((), dtype=dtype), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        pass  # variation enters via program() / snapshot(), not fabrication

    @property
    def c_top__fF(self) -> float:
        """Top-electrode (BL-side) parasitic capacitance per cell."""
        return self.config.c_top__fF

    @property
    def c_bot__fF(self) -> float:
        """Bottom-electrode (internal-node-side) parasitic capacitance per cell."""
        return self.config.c_bot__fF

    def program(self, target_g__uS: Tensor, t_elapsed: float) -> None:
        """Program the stored conductance.

        Args:
            target_g__uS: Target conductance tensor.
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
    ) -> RramSnap:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
        """
        g_view = self.g__uS.expand(shape) if shape else self.g__uS
        g = g_view if multi_coords is None else g_view[multi_coords]
        g = apply_telegraph_noise(g, self.config.read_telegraph, enabled=self.policy.read_telegraph)
        g = apply_gaussian(g, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
        g = g.clamp(self.g_min__uS, self.g_max__uS)
        return RramSnap(g__uS=g)

    def solve_dc(self, v__V: Tensor, snap: RramSnap) -> RramDcop:
        """Evaluate current and differential conductance.

        Args:
            v__V: Device voltage. Shape: arbitrary.
            snap: Conductance snap from :meth:`snapshot`.

        Returns:
            Current and local differential conductance at the requested
            voltage.
        """
        g__uS = snap.g__uS
        alpha = self.config.nonlinearity_alpha
        if alpha == 0.0:
            i__uA = g__uS * v__V
            di_dv__uS = g__uS.expand_as(i__uA)
        else:
            ax = alpha * v__V
            i__uA = g__uS * torch.sinh(ax) / alpha
            di_dv__uS = g__uS * torch.cosh(ax)
        return RramDcop(i__uA=i__uA, di_dv__uS=di_dv__uS)
