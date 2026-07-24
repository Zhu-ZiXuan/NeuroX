"""Single-ended N:1 voltage multiplexer.

See also:
    docs/reference/primitive/analog/voltage_mux.md
"""

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VmuxConfig(AnalogConfig):
    """Immutable configuration for :class:`Vmux`.

    Attributes:
        mux_ratio: N in the N:1 ratio of inputs to each output lane.
        energy_per_access__fJ: Per-access dynamic energy.
        latency_per_op__ns: Per-transport latency; multiplied by
            the runtime serial-op count at logging time.
        mux_gain: Scalar transport gain.
        mux_gain_mismatch_sigma_relative: Per-instance fractional gain
            mismatch standard deviation; flat (not area-scaled).
        mux_noise_sigma__V: Additive per-access voltage noise standard
            deviation.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    mux_ratio: int
    mux_gain: float
    mux_gain_mismatch_sigma_relative: float
    mux_noise_sigma__V: float
    energy_per_access__fJ: float
    latency_per_op__ns: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        # --- Gain and noise ---

        self._require_pos(self.mux_ratio, "mux_ratio")
        self._require_pos(self.mux_gain, "mux_gain")
        self._require_non_neg(self.mux_gain_mismatch_sigma_relative, "mux_gain_mismatch_sigma_relative")
        self._require_non_neg(self.mux_noise_sigma__V, "mux_noise_sigma__V")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_access__fJ, "energy_per_access__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class VmuxPolicy(AnalogPolicy):
    """Per-source toggles selecting which Vmux nonidealities are active.

    Attributes:
        mux_gain_mismatch: Apply ``mux_gain_mismatch_sigma_relative`` at fabricate time.
        mux_noise: Apply ``mux_noise_sigma__V`` per call.
    """

    mux_gain_mismatch: bool
    mux_noise: bool


class Vmux(AnalogBase[VmuxConfig, VmuxPolicy]):
    """Single-ended N:1 voltage transport with gain, noise, and PPA.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable PPA buffers ---

    _latency_per_op__ns: Tensor

    # --- Fabrication source buffers ---

    _nominal_eps_g: Tensor

    def __init__(
        self,
        *,
        config: VmuxConfig,
        policy: VmuxPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._sigma_eps_g = config.mux_gain_mismatch_sigma_relative
        self.register_buffer(
            "_latency_per_op__ns",
            torch.tensor(config.latency_per_op__ns, dtype=dtype),
            persistent=False,
        )
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
        v__V: Tensor,
    ) -> Tensor:
        """Transport voltages already scheduled across mux accesses and lanes.

        Args:
            v__V: Single-ended input voltages. Shape:
                ``[..., access_num, lane_num]``, where ``access_num`` equals
                ``mux_ratio``.

        Returns:
            Transported voltages with the same shape as ``v__V``.
        """
        lane_num = self.inst_shape[-1] if self.inst_shape else 1
        expected_trailing = (self.config.mux_ratio, lane_num)
        if v__V.shape[-2:] != expected_trailing:
            raise ValueError(
                f"trailing axes must be (access_num={self.config.mux_ratio}, lane_num={lane_num}); "
                f"got {tuple(v__V.shape[-2:])}"
            )

        # Shape: [*inst_shape] -> [*inst_prefix, access=1, lane_num]
        eps_g = self._eps_g.reshape(*self.inst_shape[:-1], 1, lane_num)
        gain = self.config.mux_gain * (1 + eps_g)
        v_muxed__V = gain * v__V
        v_muxed__V = apply_gaussian(
            v_muxed__V,
            self.config.mux_noise_sigma__V,
            enabled=self.policy.mux_noise,
        )

        serial_round_count = self._count_serial_rounds(v_muxed__V.numel())
        latency__ns = self._latency_per_op__ns * serial_round_count
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(v_muxed__V, self.config.energy_per_access__fJ))
        self._record_latency(latency__ns)
        return v_muxed__V
