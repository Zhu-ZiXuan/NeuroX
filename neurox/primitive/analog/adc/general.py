"""Boundary-bucketize ADC with optional Gaussian noise stages.

See also:
    docs/reference/primitive/analog/adc/general.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian
from neurox.common.quant import floor_bucketize

from .base import ADC, ADCConfig, AdcOperationPoint, ADCPolicy


@dataclass(frozen=True)
class GeneralADCConfig(ADCConfig):
    """Immutable configuration for :class:`GeneralADC`.

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
class GeneralADCPolicy(ADCPolicy):
    """Per-source toggles selecting which GeneralADC nonidealities are active.

    Attributes:
        sampling_noise: Apply ``sampling_noise__V`` at convert time.
        comparator_noise: Apply ``comparator_noise__V`` at convert time.
    """

    sampling_noise: bool
    comparator_noise: bool


@ADC.register_key(GeneralADCConfig)
class GeneralADC(ADC):
    """Boundary-bucketize ADC with two Gaussian noise stages.

    Single-mode: ``(mode, bits)`` must be ``(0, n_bits_implied_by_boundaries)``.
    """

    config: GeneralADCConfig
    boundaries: Tensor

    def __init__(
        self,
        *,
        config: GeneralADCConfig,
        policy: GeneralADCPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

        boundaries_t = torch.tensor(config.boundaries, dtype=dtype)
        if boundaries_t.numel() < 1:
            raise ValueError("GeneralADCConfig.boundaries must contain at least one threshold")
        self.register_buffer("boundaries", boundaries_t, persistent=False)

        n_codes = boundaries_t.numel() + 1
        self._n_bits = max(math.ceil(math.log2(n_codes)), 1)
        self._n_codes = n_codes
        # Topology-specific zero code: GeneralADC has fixed bit width, so its
        # midpoint code (the one representing analog 0 under a symmetric
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
        return self._n_bits

    def signed_range(self, adc_bits: int) -> tuple[int, int]:
        """Realisable signed code bounds at ``adc_bits``.

        GeneralADC's code count (``n_boundaries + 1``) is fixed at
        construction and may not equal ``2 ** adc_bits``. The actual
        signed range after the ``code - zero_code`` shift is
        ``[-zero_code, n_codes - 1 - zero_code]`` — narrower than the
        canonical SAR endpoints when ``n_codes`` is not a power of two.
        ``adc_bits`` is accepted for protocol symmetry but ignored
        because GeneralADC is single-mode by construction.
        """
        del adc_bits
        return -self._zero_code, self._n_codes - 1 - self._zero_code

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Quantise a differential analog voltage to a signed code (floor-bucketize + zero shift).

        Args:
            v_pos__V: Positive-side analog input voltage.
            v_neg__V: Negative-side analog input voltage, same shape.
            v_refs__V: Accepted for ADC-protocol symmetry and ignored —
                GeneralADC's bucketize boundaries are reference-free.
            adc_operation_point: Runtime operating point. ``adc_operation_point.adc_mode`` must be ``0``;
                ``adc_operation_point.adc_bits`` must equal the boundary-implied bit width.

        Returns:
            Signed ``int16`` code tensor in
            ``[-zero_code, n_codes - 1 - zero_code]`` (see :meth:`signed_range`),
            shaped like ``v_pos__V``.
            ``floor_bucketize`` emits an unsigned bucket index which is shifted
            by the topology-specific zero code (``n_codes // 2``, cached at
            construction) to align with the signed-output convention.
        """
        del v_refs__V  # reference-free; accepted for protocol symmetry
        self._validate_runtime_args(adc_operation_point)
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

        # GeneralADC: code shape carries no extra parallel trailing beyond
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

        # Clamp to the legal unsigned bucket range before the zero shift;
        # stochastic-rounding jitter in floor_bucketize can push values to
        # -1 or n_codes, which would skew the signed output if not bounded.
        code = code.clamp(min=0, max=self._n_codes - 1)
        return code - self._zero_code

    # --- shared helpers ---

    def _validate_runtime_args(self, adc_operation_point: AdcOperationPoint) -> None:
        if adc_operation_point.adc_mode != 0:
            raise ValueError(f"GeneralADC: mode ({adc_operation_point.adc_mode}) must be 0")
        if adc_operation_point.adc_bits != self._n_bits:
            raise ValueError(
                f"GeneralADC: bits ({adc_operation_point.adc_bits}) must equal self._n_bits ({self._n_bits})"
            )
