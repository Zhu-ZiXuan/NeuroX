"""Switched-capacitor bank for passive charge sharing.

See also:
    docs/dev/modules/analog/readout/README.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


@dataclass(frozen=True, kw_only=True)
class SwitchCapConfig(ValidateMixin):
    """Immutable physical configuration for :class:`SwitchCap`.

    Attributes:
        c_unit__fF: Unit capacitance [fF].
        cap_mismatch_sigma_relative: Per-unit-cap Pelgrom relative
            sigma. ``None`` skips mismatch.
        enable_thermal_noise: Per-cap kT/C settling-noise toggle.
        energy_per_sample_overhead__fJ: Per-bank switching overhead [fJ].
        leakage_per_inst__uW: Static leakage per bank [uW].
        area_per_inst__um2: Silicon area per bank [μm²].
        latency_per_op__ns: Settling latency per sample [ns].
    """

    c_unit__fF: float

    cap_mismatch_sigma_relative: float | None

    enable_thermal_noise: bool

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
        self._require_nonneg_or_none(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_sample_overhead__fJ, "energy_per_sample_overhead__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class SwitchCap(nn.Module, ProfiledModule):
    """Bottom-plate-sampled cap bank with passive charge-share averaging.

    Args:
        cfg: Immutable :class:`SwitchCapConfig`.
        name: Hierarchical profiler name.
        T__K: Operating temperature [K].
        dtype: Floating-point dtype.
    """

    c_unit__fF: Tensor
    c__fF: Tensor

    def __init__(
        self,
        *,
        cfg: SwitchCapConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        if not (T__K > 0.0):
            raise ValueError(f"SwitchCap.T__K ({T__K}) must be > 0")
        self.cfg = cfg
        self.T__K: float = T__K
        self.dtype = dtype

        self.register_buffer(
            "c_unit__fF",
            torch.tensor(cfg.c_unit__fF, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "c__fF",
            self.c_unit__fF.clone(),
            persistent=False,
        )

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

    def fabricate(self, shape: tuple[int, ...], cap_ratio: Tensor) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape; fabricated
                ``c__fF`` lands at ``(*shape, n_caps)``.
            cap_ratio: 1-D per-cap weight template, shape
                ``[n_caps]``.
        """
        cfg = self.cfg
        weights = cap_ratio.to(self.dtype)
        c__fF = self.c_unit__fF.clone().expand(shape).unsqueeze(-1) * weights.unsqueeze(0)
        c__fF = apply_pelgrom_mismatch(
            c__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )
        self.register_buffer("c__fF", c__fF, persistent=False)
        self._record_inst_count(shape)

    def sample_and_accumulate(self, v_in__V: Tensor) -> Tensor:
        """Sample digit voltages and run passive charge-sharing.

        Args:
            v_in__V: Per-cap sampled voltages [V],
                shape ``(*batch, *bank_shape, n_caps)``.

        Returns:
            Node voltage with shape ``(*batch, *bank_shape)``.
        """
        c__fF = self.c__fF
        if self.cfg.enable_thermal_noise:
            # kT/C settling noise: kt__fJ = k_B·T·1e15 so kt/c lands in V².
            kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
            sigma__V = torch.sqrt(kt__fJ / c__fF)
            v_hold__V = apply_gaussian(v_in__V, sigma__V)
        else:
            v_hold__V = v_in__V
        # Σ Q_k / Σ C_k, Q_k taken from the held voltage.
        c_total__fF = c__fF.sum(dim=-1)
        v_out__V = torch.sum(c__fF * v_hold__V, dim=-1) / c_total__fF

        e_caps__fJ = 0.5 * torch.sum(c__fF * v_in__V * v_in__V, dim=-1)
        dynamic_energy__fJ = e_caps__fJ + self.cfg.energy_per_sample_overhead__fJ
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return v_out__V
