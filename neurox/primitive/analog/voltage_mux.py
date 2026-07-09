"""Voltage multiplexer — differential voltage-transport behavioural block.

See also:
    docs/reference/primitive/analog/voltage_mux.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.circuit import CircuitBase, CircuitConfig
from neurox.primitive.nonideality import apply_gaussian


@dataclass(frozen=True, kw_only=True)
class VoltageMuxConfig(CircuitConfig):
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
    """

    # --- Gain ---
    mux_gain: float

    # --- Inter-leg gain mismatch ---
    mux_gain_mismatch_sigma_relative: float

    # --- Common-mode noise ---
    mux_noise_cm_sigma__V: float

    # --- Differential-mode noise ---
    mux_noise_dm_sigma__V: float

    # --- Energy / latency ---
    energy_per_access__fJ: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_gain()
        self.validate_noise()
        self.validate_ppa()

    def validate_gain(self) -> None:
        self._require_pos(self.mux_gain, "mux_gain")
        self._require_non_neg(self.mux_gain_mismatch_sigma_relative, "mux_gain_mismatch_sigma_relative")

    def validate_noise(self) -> None:
        self._require_non_neg(self.mux_noise_cm_sigma__V, "mux_noise_cm_sigma__V")
        self._require_non_neg(self.mux_noise_dm_sigma__V, "mux_noise_dm_sigma__V")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_access__fJ, "energy_per_access__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class VoltageMuxPolicy:
    """Per-source toggles selecting which VoltageMux nonidealities are active.

    Attributes:
        mux_gain_mismatch: Apply ``mux_gain_mismatch_sigma_relative`` at fabricate time.
        mux_noise_cm: Apply ``mux_noise_cm_sigma__V`` per call.
        mux_noise_dm: Apply ``mux_noise_dm_sigma__V`` per call.
    """

    mux_gain_mismatch: bool
    mux_noise_cm: bool
    mux_noise_dm: bool


class VoltageMux(CircuitBase[VoltageMuxConfig]):
    """Differential voltage-transport block — gain + CM/DM noise + access energy.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_eps_g: Tensor
    eps_g: Tensor

    def __init__(
        self,
        *,
        config: VoltageMuxConfig,
        policy: VoltageMuxPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

        # ε_g zero until fabricated; flat σ, no Pelgrom area scaling.
        self.register_buffer("nominal_eps_g", torch.zeros((), dtype=dtype), persistent=False)
        self.register_buffer("eps_g", self.nominal_eps_g.clone(), persistent=False)
        self.sigma_eps_g = config.mux_gain_mismatch_sigma_relative

    def _sample_fabricate_mismatch(self) -> None:
        """Resample inter-leg gain mismatch ε_g at ``self._inst_shape``."""
        self.eps_g = apply_gaussian(
            self.nominal_eps_g.clone().expand(self._inst_shape),
            self.sigma_eps_g,
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
        gain_pos = gain * (1 + 0.5 * self.eps_g)
        gain_neg = gain * (1 - 0.5 * self.eps_g)
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

        # VoltageMux has no extra parallel trailing beyond inst_shape;
        # serial count via the position-invariant numel rule.
        serial_op_count = max(1, v_pos__V.numel() // max(self.inst_count, 1))
        dynamic_energy__fJ = torch.full_like(v_pos__V, self.config.energy_per_access__fJ)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=v_pos__V.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)
        return v_pos_muxed__V, v_neg_muxed__V
