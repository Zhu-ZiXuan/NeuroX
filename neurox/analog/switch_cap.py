"""Switched-capacitor bank for passive charge sharing.

See also:
    docs/dev/modules/analog/readout/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin
from neurox.common.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K


@dataclass(frozen=True, kw_only=True)
class SwitchCapConfig(ValidateMixin):
    """Immutable physical configuration for :class:`SwitchCap`.

    Attributes:
        c_unit__fF: Unit capacitance [fF].
        cap_mismatch_sigma_relative: Per-unit-cap Pelgrom relative
            sigma.
        enable_cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at
            fabricate time.
        enable_sampling_thermal_noise: Apply kT/C settling noise at
            sample time.
        energy_per_sample_overhead__fJ: Per-bank switching overhead [fJ].
        leakage_per_inst__uW: Static leakage per bank [uW].
        area_per_inst__um2: Silicon area per bank [μm²].
        latency_per_op__ns: Settling latency per sample [ns].
    """

    # --- Unit capacitance ---
    c_unit__fF: float

    # --- Cap mismatch (Pelgrom) ---
    cap_mismatch_sigma_relative: float
    enable_cap_mismatch: bool

    # --- Sampling thermal noise (kT/C) ---
    enable_sampling_thermal_noise: bool

    # --- Energy / PPA ---
    energy_per_sample_overhead__fJ: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_capacitance()
        self.validate_noise()
        self.validate_ppa()

    def validate_capacitance(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")

    def validate_noise(self) -> None:
        self._require_nonneg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_sample_overhead__fJ, "energy_per_sample_overhead__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class SwitchCap(FabricateMixin, nn.Module, ProfileMixin):
    """Bottom-plate-sampled cap bank with passive charge-share averaging.

    Args:
        cfg: Immutable :class:`SwitchCapConfig`.
        name: Hierarchical profiler name.
        inst_shape: Per-instance fabrication shape.
        dtype: Floating-point dtype.
        T__K: Operating temperature [K].
        cap_weights: Per-cap multipliers on ``cfg.c_unit__fF``.
    """

    nominal_c__fF: Tensor
    c__fF: Tensor

    def __init__(
        self,
        *,
        cfg: SwitchCapConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        cap_weights: tuple[float, ...],
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        if not (T__K > 0.0):
            raise ValueError(f"SwitchCap.T__K ({T__K}) must be > 0")
        if len(cap_weights) < 1:
            raise ValueError(f"require: len(cap_weights) ({len(cap_weights)}) >= 1")
        for k, w in enumerate(cap_weights):
            if not (w > 0.0):
                raise ValueError(f"require: cap_weights[{k}] ({w}) > 0")

        self.cfg = cfg
        self._inst_shape = inst_shape
        self.T__K = T__K
        self.dtype = dtype
        self.n_caps = len(cap_weights)

        nominal_c__fF = cfg.c_unit__fF * torch.tensor(cap_weights, dtype=dtype)
        self.register_buffer("nominal_c__fF", nominal_c__fF, persistent=False)
        self.register_buffer(
            "c__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )
        self._log_static()

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per bank [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per bank [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Settling latency per sample [ns]."""
        return self.cfg.latency_per_op__ns

    def _sample_fabricate_mismatch(self) -> None:
        """Resample per-cap mismatch at ``(*self._inst_shape, n_caps)``."""
        cfg = self.cfg
        self.c__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*self._inst_shape, self.n_caps),
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
            enabled=cfg.enable_cap_mismatch,
        )

    def sample_and_accumulate(self, v_in__V: Tensor) -> Tensor:
        """Sample digit voltages and run passive charge-sharing.

        Args:
            v_in__V: Per-cap sampled voltages [V],
                shape ``(*batch, *bank_shape, n_caps)``.

        Returns:
            Node voltage with shape ``(*batch, *bank_shape)``.
        """
        c__fF = self.c__fF
        # kT/C settling noise: kt__fJ = k_B·T·1e15 so kt/c lands in V².
        kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
        sigma__V = torch.sqrt(kt__fJ / c__fF)
        v_hold__V = apply_gaussian(v_in__V, sigma__V, enabled=self.cfg.enable_sampling_thermal_noise)
        # Σ Q_k / Σ C_k, Q_k taken from the held voltage.
        c_total__fF = c__fF.sum(dim=-1)
        v_out__V = torch.sum(c__fF * v_hold__V, dim=-1) / c_total__fF

        e_caps__fJ = 0.5 * torch.sum(c__fF * v_in__V * v_in__V, dim=-1)
        dynamic_energy__fJ = e_caps__fJ + self.cfg.energy_per_sample_overhead__fJ
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return v_out__V
