"""RRAM device model with programming, read noise, and nonlinear I-V.

See also:
    docs/dev/modules/device/rram.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)
from neurox.common.validate import ValidateMixin


@dataclass(frozen=True)
class RRAMConfig(ValidateMixin):
    """Static RRAM configuration.

    Attributes:
        state_to_g_map__uS: Conductance lookup table [uS] indexed by the
            programmed discrete state.
        nonlinearity_alpha: Hyperbolic-sine I-V nonlinearity factor [1/V].
        drift_decay_rate: Power-law drift exponent.
        drift_t0: Reference drift time [s].
        c_top__fF: Top-electrode parasitic capacitance per cell [fF].
        c_bot__fF: Bottom-electrode parasitic capacitance per cell [fF].
        prog_gamma: Optional programming-variation model.
        read_telegraph: Optional telegraph-noise model.
        read_thermal: Optional Gaussian read-noise sigma [uS].
        stuck_at: Optional stuck-at fault model.
    """

    state_to_g_map__uS: list[float]

    nonlinearity_alpha: float

    drift_decay_rate: float
    drift_t0: float

    c_top__fF: float
    c_bot__fF: float

    prog_gamma: StateDependentGammaConfig | None = None

    read_telegraph: TelegraphConfig | None = None
    read_thermal: float | None = None

    stuck_at: StuckAtFaultConfig | None = None

    @property
    def num_states(self) -> int:
        """Number of available discrete conductance states."""
        return len(self.state_to_g_map__uS)

    @property
    def g_min__uS(self) -> float:
        """Minimum ideal conductance derived from state levels."""
        return self.state_to_g_map__uS[0]

    @property
    def g_max__uS(self) -> float:
        """Maximum ideal conductance derived from state levels."""
        return self.state_to_g_map__uS[-1]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_states()
        self.validate_iv()
        self.validate_drift()
        self.validate_parasitics()
        self.validate_noise()

    def validate_states(self) -> None:
        self._require_min_length(self.state_to_g_map__uS, 2, "state_to_g_map__uS")
        self._require_strictly_increasing(self.state_to_g_map__uS, "state_to_g_map__uS")

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
        self._require_nonneg_or_none(self.read_thermal, "read_thermal")


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


class RRAM(nn.Module):
    """Stateful RRAM array model."""

    state_to_g_map__uS: Tensor
    g__uS: Tensor

    def __init__(
        self,
        *,
        cfg: RRAMConfig,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Construct one stateful RRAM model.

        Args:
            cfg: RRAM configuration.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
        """
        super().__init__()

        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K
        self.g_min__uS = cfg.g_min__uS
        self.g_max__uS = cfg.g_max__uS

        self.register_buffer("state_to_g_map__uS", torch.tensor(cfg.state_to_g_map__uS, dtype=dtype), persistent=False)
        self.register_buffer("g__uS", torch.empty(0, dtype=dtype), persistent=False)

    @property
    def num_states(self) -> int:
        """Number of discrete conductance states."""
        return self.cfg.num_states

    @property
    def c_top__fF(self) -> float:
        """Top-electrode (BL-side) parasitic capacitance per cell [fF]."""
        return self.cfg.c_top__fF

    @property
    def c_bot__fF(self) -> float:
        """Bottom-electrode (Node-X-side) parasitic capacitance per cell [fF]."""
        return self.cfg.c_bot__fF

    def program(self, state: Tensor, t_elapsed: float = 0.0) -> None:
        """Program the stored conductance state.

        Args:
            state: Integer state-index tensor in `[0, num_states - 1]`.
            t_elapsed: Time elapsed since programming [s].
        """
        g__uS = self.state_to_g_map__uS[state.long()]
        g__uS = apply_state_dependent_gamma(g__uS, self.cfg.prog_gamma)
        if self.cfg.drift_decay_rate > 0.0 and t_elapsed > self.cfg.drift_t0:
            drift_factor = (t_elapsed / self.cfg.drift_t0) ** (-self.cfg.drift_decay_rate)
            g__uS = g__uS * drift_factor

        g__uS = apply_stuck_at_fault(
            x=g__uS, config=self.cfg.stuck_at, min_val=self.g_min__uS, max_val=self.g_max__uS
        )

        g__uS = g__uS.clamp(self.g_min__uS, self.g_max__uS)

        self.register_buffer("g__uS", g__uS, persistent=False)

    def snapshot(self, *, shape: tuple[int, ...]) -> RRAMSnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Snapshot shape.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        g__uS = self.g__uS.expand(shape)
        g__uS = apply_telegraph_noise(g__uS, self.cfg.read_telegraph)
        if self.cfg.read_thermal is not None:
            g__uS = apply_gaussian(g__uS, self.cfg.read_thermal)
        g__uS = g__uS.clamp(self.g_min__uS, self.g_max__uS)
        return RRAMSnapshot(g__uS=g__uS)

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
        alpha = self.cfg.nonlinearity_alpha
        if alpha == 0.0:
            i__uA = g__uS * v__V
            di_dv__uS = g__uS.expand_as(i__uA)
        else:
            ax = alpha * v__V
            i__uA = g__uS * torch.sinh(ax) / alpha
            di_dv__uS = g__uS * torch.cosh(ax)
        return RRAMDCOP(i__uA=i__uA, di_dv__uS=di_dv__uS)
