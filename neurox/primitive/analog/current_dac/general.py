"""General-purpose LUT current DAC — concrete :class:`CurrentDac` implementation.

See also:
    docs/reference/primitive/analog/current_dac/general.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import CurrentDac, CurrentDacConfig, CurrentDacPolicy


class GeneralCurrentDacConfig(CurrentDacConfig):
    """Immutable configuration for :class:`GeneralCurrentDac`.

    Attributes:
        code_to_signal: Current lookup table indexed by integer code.
            ``code_to_signal[i]`` is the nominal analog output in [uA]
            for digital code ``i``. Length equals the number of input
            codes.
        drive_thermal__uA: Signal-independent Gaussian output-noise σ added to
            each output sample after LUT lookup.
        energy_per_op__fJ: Dynamic energy per conversion operation.
        latency_per_op__ns: Per-conversion latency; multiplied by
            the runtime serial-op count at logging time.
    """

    code_to_signal: tuple[float, ...]
    drive_thermal__uA: float
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
        self._require_non_neg(self.drive_thermal__uA, "drive_thermal__uA")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class GeneralCurrentDacPolicy(CurrentDacPolicy):
    """Per-source toggles selecting which GeneralCurrentDac nonidealities are active.

    Attributes:
        drive_thermal: Apply ``drive_thermal__uA`` at convert time.
    """

    drive_thermal: bool


@CurrentDac.register_key(GeneralCurrentDacConfig)
class GeneralCurrentDac(CurrentDac[GeneralCurrentDacConfig, GeneralCurrentDacPolicy]):
    """General current DAC model — code-to-current LUT plus signal-independent output noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    code_to_signal: Tensor

    def __init__(
        self,
        *,
        config: GeneralCurrentDacConfig,
        policy: GeneralCurrentDacPolicy,
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

        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.T__K = T__K
        self.dtype = dtype

        self.register_buffer("code_to_signal", torch.tensor(config.code_to_signal, dtype=dtype), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def code_max(self) -> int:
        return len(self.config.code_to_signal) - 1

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to analog output currents.

        Args:
            code: Integer input codes.

        Returns:
            Analog output current [uA], same shape as ``code``.
        """
        signal = apply_gaussian(
            self.code_to_signal[code],
            self.config.drive_thermal__uA,
            enabled=self.policy.drive_thermal,
        )

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
