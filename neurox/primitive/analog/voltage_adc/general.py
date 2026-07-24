"""Boundary-bucketize voltage ADC with optional Gaussian noise stages.

See also:
    docs/reference/primitive/analog/voltage_adc/general.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from neurox.common.quant import floor_bucketize
from neurox.primitive.nonideality import apply_gaussian

from .base import DifferentialVoltageAdc, DifferentialVoltageAdcConfig, DifferentialVoltageAdcPolicy


@dataclass(frozen=True)
class GeneralDifferentialVoltageAdcConfig(DifferentialVoltageAdcConfig):
    """Immutable configuration for :class:`GeneralDifferentialVoltageAdc`.

    Attributes:
        boundaries: Sorted comparator thresholds in input units
            (excluding the implicit ±inf outer bounds). ``N`` thresholds
            define ``N + 1`` output codes ``[0, N]``.
        input_transform: ``"linear"`` (identity) or ``"log2"``.
        sampling_noise__V: Input-referred Gaussian sampling-stage
            noise σ.
        comparator_noise__V: Comparator (thermal/decision) noise σ
            on the signal.
        energy_per_op__fJ: Dynamic energy per conversion.
        latency_per_op__ns: Per-conversion latency; multiplied by
            the runtime serial-op count at logging time.
    """

    # --- Bucketize boundaries ---
    boundaries: tuple[float, ...]

    # --- Sampling noise ---
    sampling_noise__V: float

    # --- Comparator noise ---
    comparator_noise__V: float

    # --- Input transform ---
    input_transform: Literal["linear", "log2"]

    # --- Energy / latency ---
    energy_per_op__fJ: float
    latency_per_op__ns: float

    def validate(self) -> None:
        super().validate()
        self.validate_boundaries()
        self.validate_noise()
        self.validate_ppa()

    def validate_boundaries(self) -> None:
        self._require_min_length(self.boundaries, 1, "boundaries")
        self._require_increasing(self.boundaries, "boundaries")

    def validate_noise(self) -> None:
        self._require_non_neg(self.sampling_noise__V, "sampling_noise__V")
        self._require_non_neg(self.comparator_noise__V, "comparator_noise__V")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class GeneralDifferentialVoltageAdcPolicy(DifferentialVoltageAdcPolicy):
    """Per-source toggles selecting which GeneralDifferentialVoltageAdc nonidealities are active.

    Attributes:
        sampling_noise: Apply ``sampling_noise__V`` at convert time.
        comparator_noise: Apply ``comparator_noise__V`` at convert time.
    """

    sampling_noise: bool
    comparator_noise: bool


@DifferentialVoltageAdc.register_key(GeneralDifferentialVoltageAdcConfig)
class GeneralDifferentialVoltageAdc(
    DifferentialVoltageAdc[GeneralDifferentialVoltageAdcConfig, GeneralDifferentialVoltageAdcPolicy]
):
    """Boundary-bucketize voltage ADC with two Gaussian noise stages.

    Reference-free and mode-blind: ``bits`` must equal the
    boundary-implied bit width.
    """

    boundaries: Tensor

    def __init__(
        self,
        *,
        config: GeneralDifferentialVoltageAdcConfig,
        policy: GeneralDifferentialVoltageAdcPolicy,
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
        self.dtype = dtype
        self.T__K = T__K

        boundaries_t = torch.tensor(config.boundaries, dtype=dtype)
        if boundaries_t.numel() < 1:
            raise ValueError("GeneralDifferentialVoltageAdcConfig.boundaries must contain at least one threshold")
        self.register_buffer("boundaries", boundaries_t, persistent=False)

        n_codes = boundaries_t.numel() + 1
        self._bits = max(math.ceil(math.log2(n_codes)), 1)
        self._n_codes = n_codes
        # Topology-specific zero code: GeneralDifferentialVoltageAdc has fixed bit width, so
        # its midpoint code (the one representing analog 0 under a symmetric
        # boundary placement) is committed at construction.
        self._zero_code = n_codes // 2

        # Average spacing → stochastic-jitter LSB estimate.
        if boundaries_t.numel() >= 2:
            self._lsb_estimate = float((boundaries_t[1:] - boundaries_t[:-1]).mean().item())
        else:
            self._lsb_estimate = float(boundaries_t.item())

    def _sample_fabricate_mismatch(self) -> None:
        pass  # sampling / comparator noise are drawn per convert(), not fabricated

    # --- ADC interface ---

    @property
    def max_bits(self) -> int:
        """Boundary-implied bit width."""
        return self._bits

    @property
    def zero_code(self) -> int:
        """Raw code representing analog zero — the fixed bucket midpoint ``n_codes // 2``."""
        return self._zero_code

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Realisable raw code bounds at ``bits`` — ``(0, n_codes - 1)``.

        GeneralDifferentialVoltageAdc's code count (``n_boundaries + 1``) is fixed at
        construction and may not equal ``2 ** bits``. The raw bucket
        index ranges over ``[0, n_codes - 1]``. ``bits`` is accepted
        for protocol symmetry but ignored.
        """
        del bits
        return 0, self._n_codes - 1

    def zero_offset(self, bits: int) -> int:
        """Bucket-midpoint zero code, bit-independent (single fixed bit width)."""
        del bits
        return self._zero_code

    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_ref__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Quantise a differential analog voltage to a raw code (floor-bucketize).

        Args:
            v_pos__V: Positive-side analog input voltage.
            v_neg__V: Negative-side analog input voltage, same shape.
            v_ref__V: Accepted for ADC-protocol symmetry and ignored —
                GeneralDifferentialVoltageAdc's bucketize boundaries are reference-free.
            bits: Active resolution [bits]; must equal the boundary-implied
                bit width.

        Returns:
            Raw unsigned ``int16`` bucket-index code tensor in
            ``[0, n_codes - 1]`` (see :meth:`unsigned_range`), shaped like
            ``v_pos__V``. The zero point (:meth:`zero_offset` / :attr:`zero_code`)
            is subtracted consumer-side, not here.
        """
        del v_ref__V  # reference-free; accepted for protocol symmetry
        self._validate_runtime_args(bits)
        signal = apply_gaussian(
            v_pos__V - v_neg__V,
            self.config.sampling_noise__V,
            enabled=self.policy.sampling_noise,
        )

        if self.config.input_transform == "log2":
            signal = torch.log2(signal.clamp_min(1e-12))

        signal = apply_gaussian(
            signal,
            self.config.comparator_noise__V,
            enabled=self.policy.comparator_noise,
        )

        code = floor_bucketize(
            signal,
            self.boundaries,
            out_dtype=torch.int16,
            training=self.training,
            lsb=self._lsb_estimate,
        )

        # GeneralDifferentialVoltageAdc: code shape carries no extra parallel trailing beyond
        # inst_shape; serial count via the position-invariant numel rule
        # (total output elements / parallel multiplicity).
        serial_op_count = max(1, code.numel() // max(self.inst_count, 1))
        dynamic_energy__fJ = torch.full_like(code, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=code.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)

        # Clamp to the legal unsigned bucket range; stochastic-rounding
        # jitter in floor_bucketize can push values to -1 or n_codes, which
        # would fall outside the raw code range if not bounded. The zero
        # point is left for the consumer to subtract.
        return code.clamp(min=0, max=self._n_codes - 1)

    # --- shared helpers ---

    def _validate_runtime_args(self, bits: int) -> None:
        if bits != self._bits:
            raise ValueError(f"GeneralDifferentialVoltageAdc: bits ({bits}) must equal self._bits ({self._bits})")
