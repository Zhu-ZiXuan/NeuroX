"""Multi-output voltage reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/voltage_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VrefConfig(AnalogConfig):
    """Immutable configuration for :class:`Vref`.

    Attributes:
        v_refs__V: Nominal reference-voltage taps. Values are unordered
            and non-negative. A zero tap remains exact under relative
            noise.
        tolerance_sigma_relative: Relative per-instance initial-accuracy
            σ [dimensionless], applied multiplicatively at fabricate
            time; ``0`` leaves the exact nominal taps.
        noise_sigma_relative: Relative per-call noise σ
            [dimensionless], applied multiplicatively at snapshot time;
            ``0`` leaves the taps noise-free.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries
            all static power, including the always-on bias network that
            generates the references.
    """

    v_refs__V: tuple[float, ...]
    tolerance_sigma_relative: float
    noise_sigma_relative: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        # --- Reference bank ---

        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        for i, v in enumerate(self.v_refs__V):
            self._require_non_neg(v, f"v_refs__V[{i}]")

        # --- Noise and PPA ---

        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VrefPolicy(AnalogPolicy):
    """Per-source toggles selecting which Vref nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class VrefSnap:
    """One sampled reference snap.

    Attributes:
        v_refs__V: Actual reference-voltage taps, post
            tolerance + noise, shape ``(*inst_shape, ref_num)``.
    """

    v_refs__V: Tensor


class Vref(AnalogBase[VrefConfig, VrefPolicy]):
    """Multi-output voltage reference with static tolerance and runtime noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Fabrication source buffers ---

    _nominal_v_refs__V: Tensor

    def __init__(
        self,
        *,
        config: VrefConfig,
        policy: VrefPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "_nominal_v_refs__V",
            torch.tensor(self.config.v_refs__V, dtype=dtype),
            persistent=False,
        )

    @property
    def ref_num(self) -> int:
        """Number of reference taps sourced by this module."""
        return len(self.config.v_refs__V)

    def _sample_fabricate_mismatch(self) -> None:
        base = self._nominal_v_refs__V.expand(*self.inst_shape, self.ref_num)
        if self.policy.tolerance:
            self._v_refs__V = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self._v_refs__V = base.clone()

    def snapshot(self) -> VrefSnap:
        """Sample reference taps with per-call noise.

        Returns:
            Per-call snap carrying the actual reference-voltage taps.
        """
        v = self._v_refs__V
        v = v * (1.0 + torch.randn_like(v) * self.config.noise_sigma_relative) if self.policy.noise else v.clone()
        return VrefSnap(v_refs__V=v)
