"""Single-ended N:1 voltage multiplexer.

See Also:
    docs/reference/primitive/analog/voltage_mux.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule
from neurox.primitive.nonideality import apply_gaussian


class VmuxConfig(ConfigBase):
    # === Multiplexing ===

    mux_ratio: int
    """N in the N:1 ratio of inputs to each output lane."""
    mux_gain: float
    """Nominal transport gain, before the per-instance mismatch."""

    # === Nonidealities ===

    mux_gain_mismatch_sigma_relative: float
    """Per-instance fractional gain-mismatch σ; flat, not area-scaled."""
    mux_noise_sigma__V: float
    """σ of the additive voltage noise drawn per access."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Dynamic energy ===

    energy_per_access__fJ: float

    def validate(self) -> None:
        super().validate()

        # --- Multiplexing ---

        self._require_pos(self.mux_ratio, "mux_ratio")
        self._require_pos(self.mux_gain, "mux_gain")

        # --- Nonidealities ---

        self._require_non_neg(self.mux_gain_mismatch_sigma_relative, "mux_gain_mismatch_sigma_relative")
        self._require_non_neg(self.mux_noise_sigma__V, "mux_noise_sigma__V")

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")

        # --- Dynamic energy ---

        self._require_non_neg(self.energy_per_access__fJ, "energy_per_access__fJ")


class VmuxPolicy(PolicyBase):
    mux_gain_mismatch: bool
    """Apply `mux_gain_mismatch_sigma_relative` at fabricate time."""
    mux_noise: bool
    """Apply `mux_noise_sigma__V` per call."""


_Config = VmuxConfig
_Policy = VmuxPolicy


class Vmux(ProfileModule):
    """Single-ended N:1 voltage transport with gain, noise, and PPA."""

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_eps_g: Tensor  # Shape: []

    # === Fabricated state ===

    _eps_g: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_fabrication_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self._register_nonpersistent_buffer("_nominal_eps_g", torch.zeros((), dtype=dtype))

    def _sample_fabrication_variation(self) -> None:
        self._eps_g = apply_gaussian(
            self._nominal_eps_g.clone().expand(self.inst_shape),
            sigma=self.config.mux_gain_mismatch_sigma_relative,
            enabled=self.policy.mux_gain_mismatch,
        )

    @torch.no_grad()
    def transport(
        self,
        v__V: Tensor,
    ) -> Tensor:
        """Apply mux gain, mismatch, and noise elementwise.

        Args:
            v__V: Single-ended input voltages.

        Returns:
            Transported voltages, gained and noised per element.
        """
        gain = self.config.mux_gain * (1 + self._eps_g)
        v_muxed__V = gain * v__V
        v_muxed__V = apply_gaussian(
            v_muxed__V,
            sigma=self.config.mux_noise_sigma__V,
            enabled=self.policy.mux_noise,
        )

        if self._is_profiler_active():
            energy__fJ = torch.full((), self.config.energy_per_access__fJ, dtype=torch.float32)
            self._record_dynamic_energy(energy__fJ.expand(v_muxed__V.shape))
        return v_muxed__V
