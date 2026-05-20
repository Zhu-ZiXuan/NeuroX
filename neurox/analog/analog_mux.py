"""Analog multiplexer — differential voltage-transport behavioural block.

See also:
    docs/dev/modules/analog/analog_mux.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


@dataclass(frozen=True, kw_only=True)
class AnalogMuxConfig(ValidateMixin):
    """Immutable configuration for :class:`AnalogMux`.

    Attributes:
        energy_per_access__fJ: Per-access dynamic energy [fJ].
        mux_gain: Scalar transport gain applied to both legs.
        mux_noise_cm_sigma__V: Common-mode noise sigma [V]; same sign
            on both legs, cancels in a differential ADC.
        mux_noise_dm_sigma__V: Differential-mode noise sigma [V];
            added to ``v_pos`` and subtracted from ``v_neg``, so it
            survives a differential ADC.
        enable_mux_noise_cm: Apply ``mux_noise_cm_sigma__V`` per call.
        enable_mux_noise_dm: Apply ``mux_noise_dm_sigma__V`` per call.
        leakage_per_inst__uW: Static leakage per instance [uW].
        area_per_inst__um2: Silicon area per instance [μm²].
        latency_per_op__ns: Per-access latency [ns].
    """

    # --- Gain ---
    mux_gain: float

    # --- Common-mode noise ---
    mux_noise_cm_sigma__V: float
    enable_mux_noise_cm: bool

    # --- Differential-mode noise ---
    mux_noise_dm_sigma__V: float
    enable_mux_noise_dm: bool

    # --- Energy / PPA ---
    energy_per_access__fJ: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_gain()
        self.validate_noise()
        self.validate_ppa()

    def validate_gain(self) -> None:
        self._require_pos(self.mux_gain, "mux_gain")

    def validate_noise(self) -> None:
        self._require_nonneg(self.mux_noise_cm_sigma__V, "mux_noise_cm_sigma__V")
        self._require_nonneg(self.mux_noise_dm_sigma__V, "mux_noise_dm_sigma__V")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_access__fJ, "energy_per_access__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class AnalogMux(nn.Module, ProfiledModule):
    """Differential voltage-transport block — gain + CM/DM noise + access energy.

    Args:
        cfg: Immutable :class:`AnalogMuxConfig`.
        name: Hierarchical profiler name.
        T__K: Operating temperature [K].
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        *,
        cfg: AnalogMuxConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        self._record_inst_count(shape)

    def transport(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply gain + CM/DM transport noise to a differential pair.

        Args:
            v_pos__V: Positive-leg input voltage [V].
            v_neg__V: Negative-leg input voltage [V], broadcastable with
                ``v_pos__V``.

        Returns:
            ``(v_pos_muxed__V, v_neg_muxed__V)`` — both share ``v_pos__V``'s shape.
        """
        gain = self.cfg.mux_gain
        v_pos_muxed__V = gain * v_pos__V
        v_neg_muxed__V = gain * v_neg__V

        # CM: same sign on both legs. DM: +pos, -neg.
        zeros = torch.zeros_like(v_pos_muxed__V)
        n_cm__V = apply_gaussian(zeros, self.cfg.mux_noise_cm_sigma__V, enabled=self.cfg.enable_mux_noise_cm)
        v_pos_muxed__V = v_pos_muxed__V + n_cm__V
        v_neg_muxed__V = v_neg_muxed__V + n_cm__V

        n_dm__V = apply_gaussian(zeros, self.cfg.mux_noise_dm_sigma__V, enabled=self.cfg.enable_mux_noise_dm)
        v_pos_muxed__V = v_pos_muxed__V + n_dm__V
        v_neg_muxed__V = v_neg_muxed__V - n_dm__V

        dynamic_energy__fJ = torch.full_like(v_pos__V, self.cfg.energy_per_access__fJ)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return v_pos_muxed__V, v_neg_muxed__V
