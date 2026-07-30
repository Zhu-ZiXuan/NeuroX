"""General-purpose LUT voltage DAC — concrete :class:`Vdac` implementation.

See also:
    docs/reference/primitive/analog/voltage_dac/general.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Vdac, VdacConfig, VdacPolicy


class GeneralVdacConfig(VdacConfig):
    """Immutable configuration for :class:`GeneralVdac`.

    Attributes:
        code_to_signal: Voltage lookup table indexed by integer code.
            ``code_to_signal[i]`` is the nominal analog output in [V]
            for digital code ``i``. Length equals the number of input
            codes.
        drive_thermal__V: Gaussian thermal noise σ added to each
            output sample after LUT lookup.
        energy_per_op__fJ: Dynamic energy per output charge/discharge
            cycle (full interface-cap C*V^2); logged only for elements
            whose nominal output is nonzero — a 0 V output delivers no
            charge and logs zero.
        latency_per_op__ns: Per-conversion latency; multiplied by
            the runtime serial-op count at logging time.
    """

    code_to_signal: tuple[float, ...]
    drive_thermal__V: float
    energy_per_op__fJ: float
    latency_per_op__ns: float

    def validate(self) -> None:
        super().validate()

        self._require_min_length(self.code_to_signal, 1, "code_to_signal")
        self._require_non_neg(self.drive_thermal__V, "drive_thermal__V")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class GeneralVdacPolicy(VdacPolicy):
    """Per-source toggles selecting which GeneralVdac nonidealities are active.

    Attributes:
        drive_thermal: Apply ``drive_thermal__V`` at convert time.
    """

    drive_thermal: bool


@Vdac.register_neurox_module(config_type=GeneralVdacConfig, policy_type=GeneralVdacPolicy)
class GeneralVdac(Vdac[GeneralVdacConfig, GeneralVdacPolicy]):
    """General voltage DAC model with a code-to-voltage LUT.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Functional buffers ===

    _code_to_signal: Tensor  # Shape: [code_num]

    # === Circuit constant buffers ===

    _latency_per_op__ns: Tensor  # Shape: []

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
            "_latency_per_op__ns",
            torch.tensor(config.latency_per_op__ns, dtype=dtype),
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
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes.
                Shape: ``[...]``.

        Returns:
            Analog output voltage [V], at the same shape as ``code``.
            Shape: ``[...]``.
        """
        nominal__V = self._code_to_signal[code]
        signal = apply_gaussian(
            nominal__V,
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )

        serial_round_count = self._count_serial_rounds(signal.numel())
        latency__ns = self._latency_per_op__ns * serial_round_count
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(signal, self.config.energy_per_op__fJ, dtype=torch.float32))
        self._record_latency(latency__ns)

        return signal
