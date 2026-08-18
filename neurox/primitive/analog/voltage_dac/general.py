"""General-purpose LUT voltage DAC — concrete `Vdac` implementation.

See Also:
    docs/reference/primitive/analog/voltage_dac/general.md
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Vdac, VdacConfig, VdacPolicy


class GeneralVdacConfig(VdacConfig):
    """Immutable configuration for `GeneralVdac`."""

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

        self._require_min_length(self.code_to_signal, 1, "code_to_signal")
        self._require_non_neg(self.drive_thermal__V, "drive_thermal__V")
        if len(self.code_to_per_op_energy__fJ) != len(self.code_to_signal):
            raise ValueError(
                f"require: len(code_to_per_op_energy__fJ) ({len(self.code_to_per_op_energy__fJ)}) "
                f"== len(code_to_signal) ({len(self.code_to_signal)})"
            )
        for code, e_op__fJ in enumerate(self.code_to_per_op_energy__fJ):
            if not (math.isfinite(e_op__fJ) and e_op__fJ >= 0.0):
                raise ValueError(
                    f"require: every code_to_per_op_energy__fJ entry finite and >= 0; got {e_op__fJ} at code {code}"
                )


class GeneralVdacPolicy(VdacPolicy):
    """Per-source toggles selecting which GeneralVdac nonidealities are active."""

    drive_thermal: bool
    """Apply `drive_thermal__V` at convert time."""


@Vdac.register_neurox_module(config_type=GeneralVdacConfig, policy_type=GeneralVdacPolicy)
class GeneralVdac(Vdac[GeneralVdacConfig, GeneralVdacPolicy]):
    """General voltage DAC model with a code-to-voltage LUT."""

    # === Functional buffers ===

    _code_to_signal: Tensor  # Shape: [code_num]
    _code_to_per_op_energy__fJ: Tensor  # Shape: [code_num]

    def __init__(
        self,
        *,
        config: GeneralVdacConfig,
        policy: GeneralVdacPolicy,
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

        self.register_buffer("_code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype), persistent=False)
        self.register_buffer(
            "_code_to_per_op_energy__fJ",
            torch.tensor(config.code_to_per_op_energy__fJ, dtype=dtype),
            persistent=False,
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def convert(self, code: Tensor) -> Tensor:
        nominal__V = self._code_to_signal[code]
        signal = apply_gaussian(
            nominal__V,
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )

        if self._is_dynamic_energy_profile_active():
            # Each element costs what its own level costs, so the energy LUT
            # is gathered exactly as the signal LUT is.
            # Shape: [*code.shape]
            self._record_dynamic_energy(self._code_to_per_op_energy__fJ[code])

        return signal
