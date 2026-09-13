"""Switched-capacitor bank for passive charge sharing.

See Also:
    docs/reference/primitive/analog/switch_cap.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.primitive.physics import thermal_fluctuation_energy__fJ


class SwitchCapConfig(ConfigBase):
    c_unit__fF: float
    """Capacitance of the weight-1 cap the bank's weights multiply."""
    cap_mismatch_sigma_relative: float
    """Per-unit-cap Pelgrom relative σ."""
    energy_per_sample_overhead__fJ: float
    """Switching overhead billed once per whole-bank sample."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:

        # --- Capacitance and mismatch ---

        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_non_neg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_sample_overhead__fJ, "energy_per_sample_overhead__fJ")


class SwitchCapPolicy(PolicyBase):
    cap_mismatch: bool
    """Apply `cap_mismatch_sigma_relative` at fabricate time."""
    sampling_thermal_noise: bool
    """Apply kT/C settling noise at sample time."""


_Config = SwitchCapConfig
_Policy = SwitchCapPolicy


class SwitchCap(ModuleBase):
    """Bottom-plate-sampled cap bank with passive charge-share averaging.

    Args:
        cap_weights: Per-cap multipliers on `config.c_unit__fF`; the length
            fixes the bank's cap count.
    """

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_c__fF: Tensor  # Shape: [cap]

    # === Fabricated state ===

    _c__fF: Tensor  # Shape: [*inst_shape, cap]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        cap_weights: tuple[float, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        if len(cap_weights) < 1:
            raise ValueError(f"require: len(cap_weights) ({len(cap_weights)}) >= 1")
        for k, w in enumerate(cap_weights):
            if not (w > 0.0):
                raise ValueError(f"require: cap_weights[{k}] ({w}) > 0")

        self._cap_num = len(cap_weights)
        self._register_fabrication_buffers(dtype=dtype, cap_weights=cap_weights)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _register_fabrication_buffers(
        self,
        *,
        dtype: torch.dtype,
        cap_weights: tuple[float, ...],
    ) -> None:
        self._register_nonpersistent_buffer(
            "_nominal_c__fF",
            self.config.c_unit__fF * torch.tensor(cap_weights, dtype=dtype),
        )

    def _sample_fabrication_variation(self) -> None:
        config = self.config
        # The floor keeps a Gaussian tail from sampling a non-positive cap: the
        # kT/C sigma takes a square root of its reciprocal and the charge-share
        # denominator sums it.
        self._c__fF = apply_pelgrom_mismatch(
            self._nominal_c__fF.clone().expand(*self.inst_shape, self._cap_num),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=self.policy.cap_mismatch,
        )

    @torch.no_grad()
    def sample_and_accumulate(self, v_in__V: Tensor) -> Tensor:
        """Sample per-cap voltages and run passive charge-sharing.

        Args:
            v_in__V: Per-cap sampled voltages.
                Shape: `[..., cap]`.

        Returns:
            Charge-weighted mean the shared node settles to.
            Shape: `[...]`.
        """
        c__fF = self._c__fF
        # kT/C settling noise: fJ / fF gives V^2.
        kt__fJ = thermal_fluctuation_energy__fJ(self.T__K)
        sigma__V = torch.sqrt(kt__fJ / c__fF)
        v_hold__V = apply_gaussian(v_in__V, sigma__V, enabled=self.policy.sampling_thermal_noise)
        c_total__fF = c__fF.sum(dim=-1)
        v_out__V = torch.sum(c__fF * v_hold__V, dim=-1) / c_total__fF

        if self._is_dynamic_energy_profile_active():
            # Shape: [..., cap] -> [...]
            e_caps__fJ = 0.5 * torch.sum(c__fF * v_in__V * v_in__V, dim=-1)
            # The cap axis is already summed above; the collector sums the
            # remaining work and instance axes past the call's leading dims.
            self._record_dynamic_energy(e_caps__fJ + self.config.energy_per_sample_overhead__fJ)
        return v_out__V
