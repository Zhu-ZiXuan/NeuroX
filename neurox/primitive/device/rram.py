"""Programmable-conductance RRAM device model.

See Also:
    docs/reference/primitive/device/rram.md
"""

import torch
from torch import Tensor

from neurox.common import ConfigBase, DcopBase, DeviceBase, PolicyBase, SnapBase
from neurox.primitive.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)


class RramConfig(ConfigBase):
    g_min__uS: float
    """Minimum programmable conductance; the maximum is supplied per instance."""

    nonlinearity_alpha: float
    """Hyperbolic-sine I-V nonlinearity factor [1/V]; `0` makes the cell ohmic."""

    drift_decay_rate: float
    """Power-law drift exponent."""
    drift_t0: float
    """Reference drift time [s]; drift applies only beyond it."""

    read_thermal__uS: float
    """Gaussian read-noise σ."""

    prog_gamma: StateDependentGammaConfig

    read_telegraph: TelegraphConfig

    stuck_at: StuckAtFaultConfig

    def validate(self) -> None:

        # --- Conductance and I-V ---

        self._require_non_neg(self.g_min__uS, "g_min__uS")
        self._require_non_neg(self.nonlinearity_alpha, "nonlinearity_alpha")

        # --- Drift and noise ---

        self._require_non_neg(self.drift_decay_rate, "drift_decay_rate")
        self._require_pos(self.drift_t0, "drift_t0")
        self._require_non_neg(self.read_thermal__uS, "read_thermal__uS")


class RramPolicy(PolicyBase):
    prog_gamma: bool
    """Apply state-dependent programming Gamma at program time."""
    drift: bool
    """Apply power-law conductance drift at program time."""
    stuck_at: bool
    """Apply stuck-at faults at program time."""
    read_telegraph: bool
    """Apply telegraph noise at snapshot time."""
    read_thermal: bool
    """Apply Gaussian read noise at snapshot time."""


class RramDcop(DcopBase):
    i__uA: Tensor
    """Current through the cell at the evaluated voltage. Shape: `[...]`."""
    di_dv__uS: Tensor
    """Slope of the I-V law at the evaluated voltage. Shape: `[...]`."""


class RramSnap(SnapBase):
    g__uS: Tensor
    """Sampled per-cell conductance, read noise included. Shape: `[...]`."""


class Rram(DeviceBase[RramConfig, RramPolicy]):
    """Stateful programmable-conductance RRAM model.

    Programming variation is applied by `program()` and read variation by
    `snapshot()`; fabrication therefore owns no RRAM state.

    Args:
        g_max__uS: Maximum programmable conductance; must exceed `g_min__uS`.
    """

    # === Programmed state ===

    _g__uS: Tensor  # Shape: [*inst_shape]

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
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if not (g_max__uS > config.g_min__uS):
            raise ValueError(f"require: g_max__uS ({g_max__uS}) > config.g_min__uS ({config.g_min__uS})")

        self._g_min__uS = config.g_min__uS
        self._g_max__uS = g_max__uS

    def program(self, target_g__uS: Tensor, t_elapsed: float) -> None:
        """Program the stored conductance.

        Args:
            target_g__uS: Target conductance tensor. Its device and dtype are
                preserved in the programmed state.
            t_elapsed: Time elapsed since programming [s].
        """
        g__uS = target_g__uS.clamp(self._g_min__uS, self._g_max__uS)
        g__uS = apply_state_dependent_gamma(g__uS, self.config.prog_gamma, enabled=self.policy.prog_gamma)
        if self.policy.drift and self.config.drift_decay_rate > 0.0 and t_elapsed > self.config.drift_t0:
            drift_factor = (t_elapsed / self.config.drift_t0) ** (-self.config.drift_decay_rate)
            g__uS = g__uS * drift_factor

        g__uS = apply_stuck_at_fault(
            x=g__uS,
            config=self.config.stuck_at,
            min_val=self._g_min__uS,
            max_val=self._g_max__uS,
            enabled=self.policy.stuck_at,
        )

        g__uS = g__uS.clamp(self._g_min__uS, self._g_max__uS)

        self._g__uS = g__uS

    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
    ) -> RramSnap:
        """Sample one per-call read conductance, applying read nonidealities.

        Args:
            shape: Broadcast shape the snap's tensor field is filled at.

        Returns:
            Per-call conductance snap, clamped to the programmable range.
        """
        g = self._g__uS.expand(shape) if shape else self._g__uS
        g = apply_telegraph_noise(g, self.config.read_telegraph, enabled=self.policy.read_telegraph)
        g = apply_gaussian(g, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
        g = g.clamp(self._g_min__uS, self._g_max__uS)
        return RramSnap(g__uS=g)

    def solve_dc(self, v__V: Tensor, snap: RramSnap) -> RramDcop:
        """Evaluate current and differential conductance at the device voltage `v__V`."""
        g__uS = snap.g__uS
        alpha = self.config.nonlinearity_alpha
        # Branching on a config float, constant per instance and resolved at trace time:
        # the ohmic limit has no sinh form to take.
        if alpha == 0.0:
            i__uA = g__uS * v__V
            di_dv__uS = g__uS.expand_as(i__uA)
        else:
            ax = alpha * v__V
            i__uA = g__uS * torch.sinh(ax) / alpha
            di_dv__uS = g__uS * torch.cosh(ax)
        return RramDcop(i__uA=i__uA, di_dv__uS=di_dv__uS)
