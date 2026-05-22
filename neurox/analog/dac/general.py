"""General-purpose LUT DAC — concrete :class:`DAC` implementation.

See also:
    docs/dev/modules/analog/dac/general.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian

from .base import DAC, DACConfig


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
        enable_drive_thermal: Apply ``drive_thermal__V`` at convert time.
        energy_per_op__fJ: Dynamic energy per conversion operation
            [fJ].
        latency_per_op__ns: Conversion latency per operation [ns].
        leakage_per_inst__uW: Static leakage power per DAC instance
            [uW].
        area_per_inst__um2: Silicon area per DAC instance [um^2].
    """

    # --- LUT ---
    code_to_signal: list[float]

    # --- Drive thermal noise ---
    drive_thermal__V: float
    enable_drive_thermal: bool

    # --- Energy / PPA ---
    energy_per_op__fJ: float
    latency_per_op__ns: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float

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
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@DAC.register_key(GeneralDACConfig)
class GeneralDAC(DAC):
    """General DAC model."""

    code_to_signal: Tensor

    def __init__(
        self,
        *,
        cfg: GeneralDACConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Initialize the LUT buffer."""
        super().__init__(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype

        self.register_buffer("code_to_signal", torch.tensor(cfg.code_to_signal, dtype=dtype), persistent=False)
        self._log_static()

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.cfg.latency_per_op__ns

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog voltages.

        Args:
            code: Integer input codes.

        Returns:
            Float analog voltages [V], same shape as ``code``.
        """
        signal = apply_gaussian(
            self.code_to_signal[code],
            self.cfg.drive_thermal__V,
            enabled=self.cfg.enable_drive_thermal,
        )

        if self.cfg.energy_per_op__fJ != 0.0:
            dynamic_energy__fJ = torch.full_like(signal, self.cfg.energy_per_op__fJ, dtype=torch.float32)
            self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        else:
            self._log_dynamic(0.0, self.cfg.latency_per_op__ns)

        return signal
