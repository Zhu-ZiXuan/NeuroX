"""Analog multiplexer — differential voltage-transport behavioural block.

See also:
    docs/dev/modules/analog/analog_mux.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ProfileMixin, ValidateMixin
from neurox.common.nonideality import apply_gaussian


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
        leakage_per_inst__uW: Static leakage per instance [uW].
        area_per_inst__um2: Silicon area per instance [μm²].
        latency_per_op__ns: Per-access latency [ns].
    """

    # --- Gain ---
    mux_gain: float

    # --- Common-mode noise ---
    mux_noise_cm_sigma__V: float

    # --- Differential-mode noise ---
    mux_noise_dm_sigma__V: float

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


@dataclass(frozen=True)
class AnalogMuxPolicy:
    """Per-source toggles selecting which AnalogMux nonidealities are active.

    Attributes:
        mux_noise_cm: Apply ``mux_noise_cm_sigma__V`` per call.
        mux_noise_dm: Apply ``mux_noise_dm_sigma__V`` per call.
    """

    mux_noise_cm: bool
    mux_noise_dm: bool


class AnalogMux(FabricateMixin, nn.Module, ProfileMixin):
    """Differential voltage-transport block — gain + CM/DM noise + access energy.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    def __init__(
        self,
        *,
        config: AnalogMuxConfig,
        policy: AnalogMuxPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.config = config
        self.policy = policy
        self._inst_shape = inst_shape
        self.dtype = dtype
        self.T__K = T__K
        self._log_static()

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.config.latency_per_op__ns

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
        gain = self.config.mux_gain
        v_pos_muxed__V = gain * v_pos__V
        v_neg_muxed__V = gain * v_neg__V

        # CM: same sign on both legs. DM: +pos, -neg.
        zeros = torch.zeros_like(v_pos_muxed__V)
        n_cm__V = apply_gaussian(zeros, self.config.mux_noise_cm_sigma__V, enabled=self.policy.mux_noise_cm)
        v_pos_muxed__V = v_pos_muxed__V + n_cm__V
        v_neg_muxed__V = v_neg_muxed__V + n_cm__V

        n_dm__V = apply_gaussian(zeros, self.config.mux_noise_dm_sigma__V, enabled=self.policy.mux_noise_dm)
        v_pos_muxed__V = v_pos_muxed__V + n_dm__V
        v_neg_muxed__V = v_neg_muxed__V - n_dm__V

        dynamic_energy__fJ = torch.full_like(v_pos__V, self.config.energy_per_access__fJ)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return v_pos_muxed__V, v_neg_muxed__V
