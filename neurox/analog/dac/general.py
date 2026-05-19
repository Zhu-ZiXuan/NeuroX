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
        drive_thermal: Gaussian thermal noise sigma added to each
            output sample after LUT lookup [V].  ``None`` skips this
            noise.
        energy_per_op__fJ: Dynamic energy per conversion operation
            [fJ].
        latency_per_op__ns: Conversion latency per operation [ns].
        leakage_per_inst__uW: Static leakage power per DAC instance
            [uW].
        area_per_inst__um2: Silicon area per DAC instance [um^2].
    """

    code_to_signal: list[float]

    drive_thermal: float | None = None

    energy_per_op__fJ: float = 0.0

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0

    def validate(self) -> None:
        super().validate()
        self.validate_lut()
        self.validate_noise()
        self.validate_ppa()

    def validate_lut(self) -> None:
        self._require_min_length(self.code_to_signal, 1, "code_to_signal")

    def validate_noise(self) -> None:
        self._require_nonneg_or_none(self.drive_thermal, "drive_thermal")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@DAC.register_config(GeneralDACConfig)
class GeneralDAC(DAC):
    """General DAC model."""

    code_to_signal: Tensor

    def __init__(
        self,
        *,
        cfg: GeneralDACConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the LUT buffer."""
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype)

        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype

        self.register_buffer("code_to_signal", torch.tensor(cfg.code_to_signal, dtype=dtype), persistent=False)

    @property
    def area_per_inst__um2(self) -> float:
        """Circuit area per instance in [um2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Circuit leakage power per instance in [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per operation in [ns]."""
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        self._record_inst_count(shape)

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog voltages.

        Args:
            code: Integer input codes.

        Returns:
            Float analog voltages [V], same shape as ``code``.
        """
        signal = self.code_to_signal[code]

        if self.cfg.drive_thermal is not None:
            signal = apply_gaussian(signal, self.cfg.drive_thermal)

        if self.cfg.energy_per_op__fJ != 0.0:
            dynamic_energy__fJ = torch.full_like(signal, self.cfg.energy_per_op__fJ, dtype=torch.float32)
            self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        else:
            self._log_dynamic(0.0, self.cfg.latency_per_op__ns)

        return signal
