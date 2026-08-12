"""Programmable-conductance RRAM device model.

See also:
    docs/reference/primitive/device/rram.md
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, TensorGroupMixin
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
    """Static RRAM device configuration.

    Attributes:
        g_min__uS: Minimum programmable conductance.
        nonlinearity_alpha: Hyperbolic-sine I-V nonlinearity factor [1/V].
        drift_decay_rate: Power-law drift exponent.
        drift_t0: Reference drift time [s].
        read_thermal__uS: Gaussian read-noise σ.
        prog_gamma: Programming-variation model parameters.
        read_telegraph: Telegraph-noise model parameters.
        stuck_at: Stuck-at fault model parameters.
    """

    g_min__uS: float

    nonlinearity_alpha: float

    drift_decay_rate: float
    drift_t0: float

    read_thermal__uS: float

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
    """Per-source toggles selecting which RRAM nonidealities are active.

    Attributes:
        prog_gamma: Apply state-dependent programming Gamma at program time.
        drift: Apply power-law conductance drift at program time.
        stuck_at: Apply stuck-at faults at program time.
        read_telegraph: Apply telegraph noise at snapshot time.
        read_thermal: Apply Gaussian read noise at snapshot time.
    """

    prog_gamma: bool
    drift: bool
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
class RramSnap(TensorGroupMixin):
    """Per-call read conductance snap.

    Attributes:
        g__uS: Sampled per-cell conductance.
    """

    g__uS: Tensor


class Rram(ModuleBase[RramConfig, RramPolicy]):
    """Stateful programmable-conductance RRAM model.

    Args:
        config: Device configuration.
        policy: Nonideality policy.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        g_max__uS: Maximum programmable conductance.
    """

    is_profile_target: ClassVar[bool] = False

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

    def _sample_fabricate_mismatch(self) -> None:
        pass

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
        """Sample one per-call runtime snap over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.

        Returns:
            Per-call snap of the fabricated state.
        """
        g = self._g__uS.expand(shape) if shape else self._g__uS
        g = apply_telegraph_noise(g, self.config.read_telegraph, enabled=self.policy.read_telegraph)
        g = apply_gaussian(g, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
        g = g.clamp(self._g_min__uS, self._g_max__uS)
        return RramSnap(g__uS=g)

    def solve_dc(self, v__V: Tensor, snap: RramSnap) -> RramDcop:
        """Evaluate current and differential conductance.

        Args:
            v__V: Device voltage.
                Shape: ``[...]``.
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
