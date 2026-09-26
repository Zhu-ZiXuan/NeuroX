"""Reference-subtracting current-sense ADC."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog.current_adc import SarIadc, SarIadcConfig, SarIadcPolicy


class RsCsaIadcConfig(SarIadcConfig):
    # === Timing ===

    init__ns: float
    """Additional duration of the first bit-decision phase."""

    # === Dynamic energy ===

    energy_per_bit__fJ: float
    """Data-independent switching energy of one bit decision."""

    def validate(self) -> None:
        super().validate()

        # --- Timing ---

        self._require_non_neg(self.init__ns, "init__ns")

        # --- Dynamic energy ---

        self._require_non_neg(self.energy_per_bit__fJ, "energy_per_bit__fJ")


class RsCsaIadcPolicy(SarIadcPolicy):
    pass


_Config = RsCsaIadcConfig
_Policy = RsCsaIadcPolicy


class RsCsaIadc(SarIadc):
    """Convert currents using one injected LSB reference per conversion position.

    Construct directly with the matching config and policy, place the converter,
    and fabricate its inherited comparator state. Unlike the general SAR ladder,
    `i_refs__uA` must have exactly one trailing tap; each trial scales that tap
    by its full-width trial code. The tap's leading axes broadcast to the input
    shape. Supply a positive reference for an increasing quantization transfer.

    `active_bits` selects the number of SAR decisions. Latency adds `init__ns`
    once to the decision periods, and profiling bills `energy_per_bit__fJ` for
    each enabled decision. Output coding and enable behavior follow
    `Iadc.convert`.
    """

    config: _Config
    policy: _Policy

    def _compute_bit_dynamic_energy__fJ(
        self,
        i_in__uA: Tensor,
        *,
        i_refs__uA: Tensor,
        trial_code: Tensor,
        bit_position: int,
        enable: Tensor | None,
    ) -> Tensor:
        # Mask presence specializes during tracing; only masked costs depend on its device.
        if enable is None:
            energy__fJ = torch.full((), self.config.energy_per_bit__fJ, dtype=torch.float32)
        else:
            energy__fJ = enable.to(dtype=torch.float32) * self.config.energy_per_bit__fJ
        return energy__fJ.expand(i_in__uA.shape)

    def latency__ns(self, *, active_bits: int) -> float:
        return self.config.init__ns + super().latency__ns(active_bits=active_bits)

    def _select_reference(self, i_refs__uA: Tensor, *, trial_code: Tensor) -> Tensor:
        if i_refs__uA.shape[-1] != 1:
            raise ValueError(f"require: i_refs__uA.shape[-1] ({i_refs__uA.shape[-1]}) == 1")
        i_lsb__uA = torch.broadcast_to(i_refs__uA[..., 0], trial_code.shape)
        return trial_code * i_lsb__uA
