"""Reference-subtracting current-sense ADC."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog.current_adc import SarIadc, SarIadcConfig, SarIadcPolicy


class RsCsaIadcConfig(SarIadcConfig):
    init__ns: float
    """Additional duration of the first bit-decision phase."""
    energy_per_bit__fJ: float
    """Data-independent switching energy of one bit decision."""

    def validate(self) -> None:
        super().validate()

        self._require_non_neg(self.init__ns, "init__ns")
        self._require_non_neg(self.energy_per_bit__fJ, "energy_per_bit__fJ")


class RsCsaIadcPolicy(SarIadcPolicy):
    pass


_Config = RsCsaIadcConfig
_Policy = RsCsaIadcPolicy


class RsCsaIadc(SarIadc):
    """Reference-subtracting current-sense ADC."""

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _energy_per_bit__fJ: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)
        self._register_nonpersistent_buffer("_energy_per_bit__fJ", torch.tensor(config.energy_per_bit__fJ, dtype=dtype))

    def latency__ns(self, *, active_bits: int) -> float:
        return self.config.init__ns + super().latency__ns(active_bits=active_bits)

    def _select_reference(self, i_refs__uA: Tensor, trial_code: Tensor) -> Tensor:
        if i_refs__uA.shape[-1] != 1:
            raise ValueError(f"require: i_refs__uA.shape[-1] ({i_refs__uA.shape[-1]}) == 1")
        i_lsb__uA = torch.broadcast_to(i_refs__uA[..., 0], trial_code.shape)
        return trial_code * i_lsb__uA

    def _compute_bit_dynamic_energy__fJ(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        trial_code: Tensor,
        *,
        bit_position: int,
    ) -> Tensor:
        del i_refs__uA, trial_code, bit_position
        return self._energy_per_bit__fJ.expand_as(i_in__uA)
