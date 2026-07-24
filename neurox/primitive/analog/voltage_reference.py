"""Multi-output voltage reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/voltage_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class VoltageReferenceConfig(AnalogConfig):
    """Immutable configuration for :class:`VoltageReference`.

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
        self.validate_taps()
        self.validate_noise()
        self.validate_ppa()

    def validate_taps(self) -> None:
        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        for i, v in enumerate(self.v_refs__V):
            self._require_non_neg(v, f"v_refs__V[{i}]")

    def validate_noise(self) -> None:
        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VoltageReferencePolicy(AnalogPolicy):
    """Per-source toggles selecting which VoltageReference nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class VoltageReferenceSnap:
    """One sampled reference snap.

    Attributes:
        v_refs__V: Actual reference-voltage taps, post
            tolerance + noise, shape ``(*inst_shape, num_refs)``.
    """

    v_refs__V: Tensor


class VoltageReference(AnalogBase[VoltageReferenceConfig, VoltageReferencePolicy]):
    """Multi-output voltage reference with static tolerance and runtime noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Fabrication source buffers ---

    nominal_v_refs__V: Tensor

    def __init__(
        self,
        *,
        config: VoltageReferenceConfig,
        policy: VoltageReferencePolicy,
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
            "nominal_v_refs__V",
            torch.tensor(self.config.v_refs__V, dtype=dtype),
            persistent=False,
        )

    @property
    def num_refs(self) -> int:
        """Number of reference taps sourced by this module."""
        return len(self.config.v_refs__V)

    def _sample_fabricate_mismatch(self) -> None:
        base = self.nominal_v_refs__V.expand(*self.inst_shape, self.num_refs)
        if self.policy.tolerance:
            self.v_refs__V = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self.v_refs__V = base.clone()

    def snapshot(self) -> VoltageReferenceSnap:
        """Sample reference taps with per-call noise.

        Returns:
            Per-call snap carrying the actual reference-voltage taps.
        """
        v = self.v_refs__V
        v = v * (1.0 + torch.randn_like(v) * self.config.noise_sigma_relative) if self.policy.noise else v.clone()
        return VoltageReferenceSnap(v_refs__V=v)

    def v_ref__V(self, snap: VoltageReferenceSnap) -> Tensor:
        """Read all reference-voltage taps from a snap.

        Args:
            snap: Per-call snap returned by :meth:`snapshot`.

        Returns:
            Reference-voltage taps, shape ``(*inst_shape, num_refs)``.
        """
        return snap.v_refs__V
