"""Single-ended N:1 voltage multiplexer.

See Also:
    docs/reference/primitive/analog/voltage_mux.md
    docs/internals/primitive/analog/voltage_mux.md
"""

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VmuxConfig(AnalogConfig):
    """Immutable configuration for `Vmux`."""

    mux_ratio: int
    """N in the N:1 ratio of inputs to each output lane."""
    mux_gain: float
    """Nominal transport gain, before the per-instance mismatch."""
    mux_gain_mismatch_sigma_relative: float
    """Per-instance fractional gain-mismatch σ; flat, not area-scaled."""
    mux_noise_sigma__V: float
    """σ of the additive voltage noise drawn per access."""
    energy_per_access__fJ: float
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


class VmuxPolicy(AnalogPolicy):
    """Per-source toggles selecting which Vmux nonidealities are active."""

    mux_gain_mismatch: bool
    """Apply `mux_gain_mismatch_sigma_relative` at fabricate time."""
    mux_noise: bool
    """Apply `mux_noise_sigma__V` per call."""


class Vmux(AnalogBase[VmuxConfig, VmuxPolicy]):
    """Single-ended N:1 voltage transport with gain, noise, and PPA.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Nominal buffers ===

    _nominal_eps_g: Tensor  # Shape: []

    # === Fabricated state ===

    _eps_g: Tensor  # Shape: [*inst_shape]

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
        self._sigma_eps_g = config.mux_gain_mismatch_sigma_relative
        self._register_fabrication_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

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
            v__V: Single-ended input voltages, where `access_num` equals
                `mux_ratio` and `lane_num` is the last extent of `inst_shape`.
                Any outer instance axes broadcast to the left of the access
                axis.
                Shape: `[..., access_num, lane_num]`.

        Returns:
            Transported voltages, gained and noised per element.
            Shape: `[..., access_num, lane_num]`.

        Raises:
            ValueError: The trailing axes are not `(mux_ratio, lane_num)`.
        """
        lane_num = self.inst_shape[-1] if self.inst_shape else 1
        expected_trailing = (self.config.mux_ratio, lane_num)
        if v__V.shape[-2:] != expected_trailing:
            raise ValueError(
                f"trailing axes must be (access_num={self.config.mux_ratio}, lane_num={lane_num}); "
                f"got {tuple(v__V.shape[-2:])}"
            )

        # Shape: [*inst_shape] -> [..., access=1, lane_num]
        eps_g = self._eps_g.reshape(*self.inst_shape[:-1], 1, lane_num)
        gain = self.config.mux_gain * (1 + eps_g)
        v_muxed__V = gain * v__V
        v_muxed__V = apply_gaussian(
            v_muxed__V,
            self.config.mux_noise_sigma__V,
            enabled=self.policy.mux_noise,
        )

        if self._is_dynamic_energy_profile_active():
            # Shape: [] -> [*v_muxed__V.shape]
            e_access__fJ = torch.full(
                (), self.config.energy_per_access__fJ, dtype=torch.float32, device=v_muxed__V.device
            )
            self._record_dynamic_energy(e_access__fJ.expand(v_muxed__V.shape))
        return v_muxed__V
