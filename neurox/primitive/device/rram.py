"""Programmable-conductance RRAM device model.

See Also:
    docs/reference/primitive/device/rram.md
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, DcopBase, ModuleBase, PolicyBase, SnapBase
from neurox.primitive.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)

__all__ = [
    "Rram",
    "RramConfig",
    "RramDcop",
    "RramPolicy",
    "RramSnap",
]


class RramConfig(ConfigBase):
    g_min__uS: float
    """Strictly positive programmable floor; the ceiling is supplied per instance."""

    nonlinearity_alpha: float
    """Hyperbolic-sine I-V nonlinearity factor [1/V]; `0` makes the cell ohmic."""

    drift_decay_rate: float
    """Power-law drift exponent."""
    drift_t0: float
    """Reference time [s] for the retained drift law."""

    read_thermal__uS: float
    """Gaussian read-noise σ."""

    prog_gamma: StateDependentGammaConfig

    read_telegraph: TelegraphConfig

    stuck_at: StuckAtFaultConfig

    def validate(self) -> None:

        # --- Conductance and I-V ---

        self._require_pos(self.g_min__uS, "g_min__uS")
        self._require_non_neg(self.nonlinearity_alpha, "nonlinearity_alpha")

        # --- Drift and noise ---

        self._require_non_neg(self.drift_decay_rate, "drift_decay_rate")
        self._require_pos(self.drift_t0, "drift_t0")
        self._require_non_neg(self.read_thermal__uS, "read_thermal__uS")


class RramPolicy(PolicyBase):
    prog_gamma: bool
    """Apply state-dependent programming Gamma at program time."""
    drift: bool
    """Drift selection; programming at time zero applies no drift."""
    stuck_at: bool
    """Apply stuck-at faults at program time."""
    read_telegraph: bool
    """Apply telegraph noise at snapshot time."""
    read_thermal: bool
    """Apply Gaussian read noise at snapshot time."""


class RramDcop(DcopBase):
    i__uA: Tensor
    """Current through the cell at the evaluated voltage."""
    di_dv__uS: Tensor
    """Slope of the I-V law at the evaluated voltage."""


class RramSnap(SnapBase):
    g__uS: Tensor
    """Sampled per-cell conductance, read noise included."""


_Config = RramConfig
_Policy = RramPolicy
_Dcop = RramDcop
_Snap = RramSnap


class Rram(ModuleBase):
    """Stateful programmable-conductance RRAM model.

    Programming variation is applied by `program()` and read variation by
    `snapshot()`; fabrication therefore owns no RRAM state.

    Args:
        g_max__uS: Maximum programmable conductance; both bounds must be
            normal values representable by `dtype`.
    """

    is_profile_target: ClassVar[bool] = False

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _g__uS: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        g_max__uS: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if not dtype.is_floating_point:
            raise TypeError(f"Rram requires a floating-point dtype; got {dtype}")
        dtype_info = torch.finfo(dtype)
        if not (dtype_info.tiny <= config.g_min__uS <= dtype_info.max):
            raise ValueError(f"config.g_min__uS ({config.g_min__uS}) is not a normal value representable by {dtype}")
        if not (config.g_min__uS < g_max__uS <= dtype_info.max):
            raise ValueError(
                f"require: config.g_min__uS ({config.g_min__uS}) < g_max__uS ({g_max__uS}) <= {dtype_info.max}"
            )

        self._g_max__uS = g_max__uS

    @torch.no_grad()
    def program(self, target_g__uS: Tensor) -> None:
        """Program the stored conductance with elapsed time fixed to zero.

        Args:
            target_g__uS: Target conductance tensor. Its device and dtype are
                preserved in the programmed state.
        """
        g_min__uS = self.config.g_min__uS
        g__uS = target_g__uS.clamp(g_min__uS, self._g_max__uS)
        g__uS = apply_state_dependent_gamma(g__uS, self.config.prog_gamma, enabled=self.policy.prog_gamma)
        # if self.policy.drift and self.config.drift_decay_rate > 0.0 and t_elapsed > self.config.drift_t0:
        #     drift_factor = (t_elapsed / self.config.drift_t0) ** (-self.config.drift_decay_rate)
        #     g__uS = g__uS * drift_factor

        g__uS = apply_stuck_at_fault(
            x=g__uS,
            config=self.config.stuck_at,
            min_val=g_min__uS,
            max_val=self._g_max__uS,
            enabled=self.policy.stuck_at,
        )

        g__uS = g__uS.clamp(g_min__uS, self._g_max__uS)

        self._g__uS = g__uS

    @torch.no_grad()
    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
    ) -> _Snap:
        """Sample one per-call read conductance, applying read nonidealities.

        Args:
            shape: Broadcast shape the snap's tensor field is filled at.

        Returns:
            Per-call conductance snap, clamped to the programmable range.
        """
        g = self._g__uS.expand(shape) if shape else self._g__uS
        g = apply_telegraph_noise(g, self.config.read_telegraph, enabled=self.policy.read_telegraph)
        g = apply_gaussian(g, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
        g = g.clamp(self.config.g_min__uS, self._g_max__uS)
        return _Snap(g__uS=g)

    @torch.no_grad()
    def solve_dc(self, v__V: Tensor, snap: _Snap) -> _Dcop:
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
        return _Dcop(i__uA=i__uA, di_dv__uS=di_dv__uS)
