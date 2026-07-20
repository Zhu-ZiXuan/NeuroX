"""General-purpose LUT voltage DAC — concrete :class:`VoltageDac` implementation.

See also:
    docs/reference/primitive/analog/voltage_dac/general.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import VoltageDac, VoltageDacConfig, VoltageDacPolicy


@dataclass(frozen=True)
class GeneralVoltageDacConfig(VoltageDacConfig):
    """Immutable configuration for :class:`GeneralVoltageDac`.

    Attributes:
        code_to_signal: Voltage lookup table indexed by integer code.
            ``code_to_signal[i]`` is the nominal analog output in [V]
            for digital code ``i``. Length equals the number of input
            codes.
        drive_thermal__V: Gaussian thermal noise σ [V] added to each
            output sample after LUT lookup.
        energy_per_op__fJ: Dynamic energy per output charge/discharge
            cycle (full interface-cap C*V^2); logged only for elements
            whose nominal output is nonzero — a 0 V output delivers no
            charge and logs zero.
        latency_per_op__ns: Per-conversion latency; multiplied by
            the runtime serial-op count at logging time.
    """

    # --- LUT ---
    code_to_signal: tuple[float, ...]

    # --- Drive thermal noise ---
    drive_thermal__V: float

    # --- Energy / latency ---
    energy_per_op__fJ: float
    latency_per_op__ns: float

    def validate(self) -> None:
        super().validate()
        self.validate_lut()
        self.validate_noise()
        self.validate_ppa()

    def validate_lut(self) -> None:
        self._require_min_length(self.code_to_signal, 1, "code_to_signal")

    def validate_noise(self) -> None:
        self._require_non_neg(self.drive_thermal__V, "drive_thermal__V")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class GeneralVoltageDacPolicy(VoltageDacPolicy):
    """Per-source toggles selecting which GeneralVoltageDac nonidealities are active.

    Attributes:
        drive_thermal: Apply ``drive_thermal__V`` at convert time.
    """

    drive_thermal: bool


@VoltageDac.register_key(GeneralVoltageDacConfig)
class GeneralVoltageDac(VoltageDac):
    """General voltage DAC model — code-to-voltage LUT plus output thermal noise."""

    config: GeneralVoltageDacConfig
    policy: GeneralVoltageDacPolicy
    code_to_signal: Tensor

    def __init__(
        self,
        *,
        config: GeneralVoltageDacConfig,
        policy: GeneralVoltageDacPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Initialize the LUT buffer."""
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.T__K = T__K
        self.dtype = dtype

        self.register_buffer("code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        pass  # drive-thermal noise is drawn per convert(), not fabricated

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output voltages.

        Args:
            code: Integer input codes.

        Returns:
            Analog output voltage [V], same shape as ``code``.
        """
        nominal__V = self.code_to_signal[code]
        signal = apply_gaussian(
            nominal__V,
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )

        # Serial-op count via the position-invariant numel rule: total
        # output elements / parallel hardware multiplicity. For DAC the
        # parallel structure is exactly ``inst_count`` — no extra
        # parallel trailing — so the divisor is ``self.inst_count``.
        serial_op_count = max(1, signal.numel() // max(self.inst_count, 1))
        # Per-op driver-circuit energy: one constant per conversion op
        # (the drive LOAD's capacitive cycling is billed by the load's
        # owner, e.g. the array's WL wire + gate terms — not here).
        dynamic_energy__fJ = torch.full_like(signal, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=signal.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)

        return signal
