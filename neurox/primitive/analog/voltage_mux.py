"""Voltage multiplexer — differential voltage-transport behavioural block.

See also:
    docs/reference/primitive/analog/voltage_mux.md
"""

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy
from neurox.primitive.nonideality import apply_gaussian


class VoltageMuxConfig(AnalogConfig):
    """Immutable configuration for :class:`VoltageMux`.

    Attributes:
        energy_per_access__fJ: Per-access dynamic energy.
        latency_per_op__ns: Per-transport latency; multiplied by
            the runtime serial-op count at logging time.
        mux_gain: Scalar matched transport gain applied to both legs.
        mux_gain_mismatch_sigma_relative: Per-mux fractional inter-leg
            gain-mismatch σ; flat (not area-scaled).
        mux_noise_cm_sigma__V: Common-mode noise σ; same sign
            on both legs, cancels in a differential ADC.
        mux_noise_dm_sigma__V: Differential-mode noise σ;
            added to ``v_pos`` and subtracted from ``v_neg``, so it
            survives a differential ADC.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    mux_gain: float
    mux_gain_mismatch_sigma_relative: float
    mux_noise_cm_sigma__V: float
    mux_noise_dm_sigma__V: float
    energy_per_access__fJ: float
    latency_per_op__ns: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        # --- Gain and noise ---

        self._require_pos(self.mux_gain, "mux_gain")
        self._require_non_neg(self.mux_gain_mismatch_sigma_relative, "mux_gain_mismatch_sigma_relative")
        self._require_non_neg(self.mux_noise_cm_sigma__V, "mux_noise_cm_sigma__V")
        self._require_non_neg(self.mux_noise_dm_sigma__V, "mux_noise_dm_sigma__V")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_access__fJ, "energy_per_access__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class VoltageMuxPolicy(AnalogPolicy):
    """Per-source toggles selecting which VoltageMux nonidealities are active.

    Attributes:
        mux_gain_mismatch: Apply ``mux_gain_mismatch_sigma_relative`` at fabricate time.
        mux_noise_cm: Apply ``mux_noise_cm_sigma__V`` per call.
        mux_noise_dm: Apply ``mux_noise_dm_sigma__V`` per call.
    """

    mux_gain_mismatch: bool
    mux_noise_cm: bool
    mux_noise_dm: bool


class VoltageMux(AnalogBase[VoltageMuxConfig, VoltageMuxPolicy]):
    """Differential voltage-transport block — gain + CM/DM noise + access energy.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Fabrication source buffers ---

    _nominal_eps_g: Tensor

    def __init__(
        self,
        *,
        config: VoltageMuxConfig,
        policy: VoltageMuxPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._sigma_eps_g = config.mux_gain_mismatch_sigma_relative
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer("_nominal_eps_g", torch.zeros((), dtype=dtype), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        self._eps_g = apply_gaussian(
            self._nominal_eps_g.clone().expand(self.inst_shape),
            self._sigma_eps_g,
            enabled=self.policy.mux_gain_mismatch,
        )

    def transport(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply gain + CM/DM transport noise to a differential pair.

        Args:
            v_pos__V: Positive-leg input voltage.
            v_neg__V: Negative-leg input voltage, broadcastable with
                ``v_pos__V``.

        Returns:
            ``(v_pos_muxed__V, v_neg_muxed__V)`` — both share ``v_pos__V``'s shape.
        """
        gain = self.config.mux_gain
        gain_pos = gain * (1 + 0.5 * self._eps_g)
        gain_neg = gain * (1 - 0.5 * self._eps_g)
        v_pos_muxed__V = gain_pos * v_pos__V
        v_neg_muxed__V = gain_neg * v_neg__V

        # CM: same sign on both legs. DM: +pos, -neg.
        zeros = torch.zeros_like(v_pos_muxed__V)
        n_cm__V = apply_gaussian(zeros, self.config.mux_noise_cm_sigma__V, enabled=self.policy.mux_noise_cm)
        v_pos_muxed__V = v_pos_muxed__V + n_cm__V
        v_neg_muxed__V = v_neg_muxed__V + n_cm__V

        n_dm__V = apply_gaussian(zeros, self.config.mux_noise_dm_sigma__V, enabled=self.policy.mux_noise_dm)
        v_pos_muxed__V = v_pos_muxed__V + n_dm__V
        v_neg_muxed__V = v_neg_muxed__V - n_dm__V

        serial_op_count = max(1, v_pos__V.numel() // max(self.inst_count, 1))
        dynamic_energy__fJ = torch.full_like(v_pos__V, self.config.energy_per_access__fJ)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=v_pos__V.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._record_dynamic_energy(dynamic_energy__fJ)
        self._record_latency(latency__ns)
        return v_pos_muxed__V, v_neg_muxed__V
