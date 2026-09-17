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
    # === Transfer ===

    code_to_signal: tuple[float, ...]
    """Current lookup table indexed by integer code: entry `i` is the nominal
    analog output [uA] for digital code `i`, and the length fixes the code
    count."""

    # === Nonidealities ===

    drive_thermal__uA: float
    """Signal-independent Gaussian output-noise σ added to each output sample
    after LUT lookup."""

    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy of converting one element, flat across codes."""

    def validate(self) -> None:
        super().validate()

        # --- Transfer ---

        self._require_non_empty(self.code_to_signal, "code_to_signal")

        # --- Nonidealities ---

        self._require_non_neg(self.drive_thermal__uA, "drive_thermal__uA")

        # --- Dynamic energy ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class GeneralIdacPolicy(IdacPolicy):
    drive_thermal: bool
    """Apply `drive_thermal__uA` at convert time."""


_Config = GeneralIdacConfig
_Policy = GeneralIdacPolicy


@Idac.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class GeneralIdac(Idac):
    """General current DAC model — code-to-current LUT plus signal-independent output noise."""

    config: _Config
    policy: _Policy

    _code_to_signal: Tensor  # Shape: [code]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

        self._register_nonpersistent_buffer("_code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype))

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def _convert_impl(self, code: Tensor) -> Tensor:
        signal = apply_gaussian(
            self._code_to_signal[code.long()],
            self.config.drive_thermal__uA,
            enabled=self.policy.drive_thermal,
        )

        if self._is_profiler_active():
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=signal.device)
            self._record_dynamic_energy(e_op__fJ.expand(signal.shape))

        return signal
