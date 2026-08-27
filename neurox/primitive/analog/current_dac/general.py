"""General-purpose LUT current DAC — concrete `Idac` implementation.

See Also:
    docs/reference/primitive/analog/current_dac/general.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Idac, IdacConfig, IdacPolicy


class GeneralIdacConfig(IdacConfig):
    code_to_signal: tuple[float, ...]
    """Current lookup table indexed by integer code: entry `i` is the nominal
    analog output [uA] for digital code `i`, and the length fixes the code
    count."""
    drive_thermal__uA: float
    """Signal-independent Gaussian output-noise σ added to each output sample
    after LUT lookup."""
    energy_per_op__fJ: float
    """Dynamic energy of converting one element, flat across codes."""

    def validate(self) -> None:
        super().validate()

        self._require_non_empty(self.code_to_signal, "code_to_signal")
        self._require_non_neg(self.drive_thermal__uA, "drive_thermal__uA")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class GeneralIdacPolicy(IdacPolicy):
    drive_thermal: bool
    """Apply `drive_thermal__uA` at convert time."""


@Idac.register_neurox_module(config_type=GeneralIdacConfig, policy_type=GeneralIdacPolicy)
class GeneralIdac(Idac[GeneralIdacConfig, GeneralIdacPolicy]):
    """General current DAC model — code-to-current LUT plus signal-independent output noise."""

    _code_to_signal: Tensor  # Shape: [code_num]

    def __init__(
        self,
        *,
        config: GeneralIdacConfig,
        policy: GeneralIdacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self._register_nonpersistent_buffer("_code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype))

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def convert(self, code: Tensor) -> Tensor:
        signal = apply_gaussian(
            self._code_to_signal[code],
            self.config.drive_thermal__uA,
            enabled=self.policy.drive_thermal,
        )

        if self._is_dynamic_energy_profile_active():
            # Shape: [] -> [*signal.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=signal.device)
            self._record_dynamic_energy(e_op__fJ.expand(signal.shape))

        return signal
