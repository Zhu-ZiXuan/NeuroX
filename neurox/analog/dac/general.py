"""General-purpose LUT DAC — concrete :class:`DAC` implementation.

See also:
    docs/dev/modules/analog/dac/general.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian

from .base import DAC, DACConfig, DACPolicy


@dataclass(frozen=True)
class GeneralDACConfig(DACConfig):
    """Immutable configuration for :class:`GeneralDAC`.

    Attributes:
        code_to_signal: Voltage lookup table indexed by integer code.
            ``code_to_signal[i]`` is the nominal analog output in [V]
            for digital code ``i``.  Length equals the number of input
            codes.
        drive_thermal__V: Gaussian thermal noise sigma added to each
            output sample after LUT lookup [V].
        energy_per_op__fJ: Dynamic energy per conversion operation
            [fJ].
        latency_per_op__ns: Per-conversion latency [ns]; multiplied by
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
        self._require_nonneg(self.drive_thermal__V, "drive_thermal__V")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class GeneralDACPolicy(DACPolicy):
    """Per-source toggles selecting which GeneralDAC nonidealities are active.

    Attributes:
        drive_thermal: Apply ``drive_thermal__V`` at convert time.
    """

    drive_thermal: bool


@DAC.register_key(GeneralDACConfig)
class GeneralDAC(DAC):
    """General DAC model."""

    config: GeneralDACConfig
    code_to_signal: Tensor

    def __init__(
        self,
        *,
        config: GeneralDACConfig,
        policy: GeneralDACPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Initialize the LUT buffer."""
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype

        self.register_buffer("code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype), persistent=False)

    @property
    def code_max(self) -> int:
        """Maximum valid input code (inclusive); valid codes lie in ``[0, code_max]``."""
        return len(self.config.code_to_signal) - 1

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog voltages.

        Args:
            code: Integer input codes.

        Returns:
            Float analog voltages [V], same shape as ``code``.
        """
        signal = apply_gaussian(
            self.code_to_signal[code],
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )

        # Serial-op count via the position-invariant numel rule: total
        # output elements / parallel hardware multiplicity. For DAC the
        # parallel structure is exactly ``inst_count`` — no extra
        # parallel trailing — so the divisor is ``self.inst_count``.
        serial_op_count = max(1, signal.numel() // max(self.inst_count, 1))
        dynamic_energy__fJ = torch.full_like(signal, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=signal.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)

        return signal
