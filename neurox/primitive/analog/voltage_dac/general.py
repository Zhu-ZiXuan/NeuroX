"""General-purpose LUT voltage DAC — concrete `Vdac` implementation.

See Also:
    docs/reference/primitive/analog/voltage_dac/general.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Vdac, VdacConfig, VdacPolicy


class GeneralVdacConfig(VdacConfig):
    code_to_signal: tuple[float, ...]
    """Voltage lookup table indexed by integer code: entry `i` is the nominal
    analog output [V] for digital code `i`, and the length fixes the code
    count."""
    drive_thermal__V: float
    """Gaussian thermal noise σ added to each output sample after LUT
    lookup."""
    code_to_per_op_energy__fJ: tuple[float, ...]
    """Dynamic energy of converting ONE element, indexed by that element's
    code — parallel to `code_to_signal`, so each level states what driving it
    costs. Same length as `code_to_signal`; every entry finite and >= 0."""

    def validate(self) -> None:
        super().validate()

        self._require_non_empty(self.code_to_signal, "code_to_signal")
        self._require_non_neg(self.drive_thermal__V, "drive_thermal__V")
        self._require_same_len(
            self.code_to_per_op_energy__fJ,
            "code_to_per_op_energy__fJ",
            self.code_to_signal,
            "code_to_signal",
        )
        for code, e_op__fJ in enumerate(self.code_to_per_op_energy__fJ):
            self._require_non_neg(e_op__fJ, f"code_to_per_op_energy__fJ[{code}]")


class GeneralVdacPolicy(VdacPolicy):
    drive_thermal: bool
    """Apply `drive_thermal__V` at convert time."""


_Config = GeneralVdacConfig
_Policy = GeneralVdacPolicy


@Vdac.register_impl(config_type=_Config, policy_type=_Policy)
class GeneralVdac(Vdac):
    """General voltage DAC model with a code-to-voltage LUT."""

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _code_to_signal: Tensor  # Shape: [code]
    _code_to_per_op_energy__fJ: Tensor  # Shape: [code]

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
        self._register_nonpersistent_buffer(
            "_code_to_per_op_energy__fJ",
            torch.tensor(config.code_to_per_op_energy__fJ, dtype=dtype),
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def _convert_impl(self, code: Tensor) -> Tensor:
        nominal__V = self._code_to_signal[code.long()]
        signal = apply_gaussian(
            nominal__V,
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )

        if self._is_dynamic_energy_profile_active():
            # Each element costs what its own level costs, so the energy LUT
            # is gathered exactly as the signal LUT is.
            self._record_dynamic_energy(self._code_to_per_op_energy__fJ[code.long()])

        return signal
