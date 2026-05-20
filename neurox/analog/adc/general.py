"""Boundary-bucketize ADC with optional Gaussian noise stages.

See also:
    docs/dev/modules/analog/adc/general.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.common.quant import floor_bucketize

from .base import ADC, ADCConfig


@dataclass(frozen=True)
class GeneralADCConfig(ADCConfig):
    """Immutable configuration for :class:`GeneralADC`.

    Attributes:
        boundaries: Sorted comparator thresholds in input units
            (excluding the implicit ±∞ outer bounds). ``N`` thresholds
            define ``N + 1`` output codes ``[0, N]``.
        drive_value: BL-clamp reference voltage [V].
        input_transform: ``"linear"`` (identity) or ``"log2"``.
        sampling_noise__V: Input-referred Gaussian sampling-stage
            noise sigma [V].
        comparator_noise__V: Per-comparator threshold offset noise
            sigma [V].
        drive_thermal__V: Gaussian thermal noise sigma on the drive
            output [V].
        enable_sampling_noise: Apply ``sampling_noise__V`` at convert
            time.
        enable_comparator_noise: Apply ``comparator_noise__V`` at
            convert time.
        enable_drive_thermal: Apply ``drive_thermal__V`` at drive time.
        energy_per_op__fJ: Dynamic energy per conversion.
        latency_per_op__ns: Conversion latency.
        leakage_per_inst__uW: Static leakage power per instance.
        area_per_inst__um2: Silicon area per instance.
    """

    # --- Bucketize boundaries ---
    boundaries: list[float]

    # --- Sampling noise ---
    sampling_noise__V: float
    enable_sampling_noise: bool

    # --- Comparator noise ---
    comparator_noise__V: float
    enable_comparator_noise: bool

    # --- Drive thermal noise ---
    drive_thermal__V: float
    enable_drive_thermal: bool

    # --- Drive reference + input transform ---
    drive_value: float = 0.0
    input_transform: Literal["linear", "log2"] = "linear"

    # --- Energy / PPA ---
    energy_per_op__fJ: float = 0.0
    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0

    def validate(self) -> None:
        super().validate()
        self.validate_boundaries()
        self.validate_noise()
        self.validate_ppa()

    def validate_boundaries(self) -> None:
        self._require_min_length(self.boundaries, 1, "boundaries")
        self._require_strictly_increasing(self.boundaries, "boundaries")

    def validate_noise(self) -> None:
        self._require_nonneg(self.sampling_noise__V, "sampling_noise__V")
        self._require_nonneg(self.comparator_noise__V, "comparator_noise__V")
        self._require_nonneg(self.drive_thermal__V, "drive_thermal__V")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@ADC.register_key(GeneralADCConfig)
class GeneralADC(ADC):
    """Boundary-bucketize ADC with three Gaussian noise stages.

    Single-mode: ``(mode, bits)`` must be ``(0, n_bits_implied_by_boundaries)``.
    """

    boundaries: Tensor
    drive_value: Tensor

    def __init__(
        self,
        *,
        cfg: GeneralADCConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        stochastic: bool | None,
    ) -> None:
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype, stochastic=stochastic)
        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K
        self.stochastic = stochastic

        boundaries_t = torch.tensor(cfg.boundaries, dtype=dtype)
        if boundaries_t.numel() < 1:
            raise ValueError("GeneralADCConfig.boundaries must contain at least one threshold")
        self.register_buffer("boundaries", boundaries_t, persistent=False)

        self.register_buffer("drive_value", torch.tensor(cfg.drive_value, dtype=dtype), persistent=False)

        n_codes = boundaries_t.numel() + 1
        self._n_bits = max(math.ceil(math.log2(n_codes)), 1)

        # Average spacing → stochastic-jitter LSB estimate.
        if boundaries_t.numel() >= 2:
            self._lsb_estimate = float((boundaries_t[1:] - boundaries_t[:-1]).mean().item())
        else:
            self._lsb_estimate = float(boundaries_t.item())

    # --- ADC interface ---

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    def latency_per_op__ns(self, *, bits: int) -> float:
        if bits != self._n_bits:
            raise ValueError(f"GeneralADC: bits ({bits}) must equal self._n_bits ({self._n_bits})")
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        self._record_inst_count(shape)

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        mode: int,
        bits: int,
    ) -> Tensor:
        """Quantise a differential analog voltage to an integer code (floor-bucketize).

        Args:
            v_pos__V: Positive-side analog input voltage [V].
            v_neg__V: Negative-side analog input voltage [V], same shape.
            mode: Operating-point index; must be ``0``.
            bits: Active bit width; must equal the boundary-implied bit width.

        Returns:
            ``int16`` code tensor shaped like ``v_pos__V``.
        """
        self._validate_runtime_args(mode, bits)
        signal = apply_gaussian(
            v_pos__V - v_neg__V,
            self.cfg.sampling_noise__V,
            enabled=self.cfg.enable_sampling_noise,
        )

        if self.cfg.input_transform == "log2":
            signal = torch.log2(signal.clamp_min(1e-12))

        signal = apply_gaussian(
            signal,
            self.cfg.comparator_noise__V,
            enabled=self.cfg.enable_comparator_noise,
        )

        code = floor_bucketize(
            signal,
            self.boundaries,
            out_dtype=torch.int16,
            training=self.training,
            override=self.stochastic,
            lsb=self._lsb_estimate,
        )

        dynamic_energy__fJ = torch.full_like(code, self.cfg.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return code

    # --- drive() shim ---

    def drive(self, shape: tuple[int, ...]) -> Tensor:
        """BL-clamp reference voltage broadcast.

        Args:
            shape: Output shape.

        Returns:
            ``drive_value`` broadcast to ``shape``, with optional Gaussian
            thermal noise.
        """
        return apply_gaussian(
            self.drive_value.expand(shape),
            self.cfg.drive_thermal__V,
            enabled=self.cfg.enable_drive_thermal,
        )

    # --- shared helpers ---

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        if mode != 0:
            raise ValueError(f"GeneralADC: mode ({mode}) must be 0")
        if bits != self._n_bits:
            raise ValueError(f"GeneralADC: bits ({bits}) must equal self._n_bits ({self._n_bits})")
