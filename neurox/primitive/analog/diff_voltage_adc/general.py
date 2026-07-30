"""Boundary-bucketize voltage ADC with optional Gaussian noise stages.

See also:
    docs/reference/primitive/analog/diff_voltage_adc/general.md
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from neurox.common.quant import floor_bucketize
from neurox.primitive.nonideality import apply_gaussian

from .base import DiffVadc, DiffVadcConfig, DiffVadcPolicy


class GeneralDiffVadcConfig(DiffVadcConfig):
    """Immutable configuration for :class:`GeneralDiffVadc`.

    Attributes:
        boundaries: Sorted comparator thresholds in input units
            (excluding the implicit ±inf outer bounds). ``N`` thresholds
            define ``N + 1`` output codes ``[0, N]``.
        sampling_noise__V: Input-referred Gaussian sampling-stage
            noise σ.
        comparator_noise__V: Comparator (thermal/decision) noise σ
            on the signal.
        energy_per_op__fJ: Dynamic energy per conversion.
        latency_per_op__ns: Per-conversion latency; multiplied by
            the runtime serial-op count at logging time.
    """

    boundaries: tuple[float, ...]
    sampling_noise__V: float
    comparator_noise__V: float
    energy_per_op__fJ: float
    latency_per_op__ns: float

    def validate(self) -> None:
        super().validate()

        # --- Transfer ---

        self._require_min_length(self.boundaries, 1, "boundaries")
        self._require_increasing(self.boundaries, "boundaries")

        # --- Noise and PPA ---

        self._require_non_neg(self.sampling_noise__V, "sampling_noise__V")
        self._require_non_neg(self.comparator_noise__V, "comparator_noise__V")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class GeneralDiffVadcPolicy(DiffVadcPolicy):
    """Per-source toggles selecting which GeneralDiffVadc nonidealities are active.

    Attributes:
        sampling_noise: Apply ``sampling_noise__V`` at convert time.
        comparator_noise: Apply ``comparator_noise__V`` at convert time.
    """

    sampling_noise: bool
    comparator_noise: bool


@DiffVadc.register_neurox_module(
    config_type=GeneralDiffVadcConfig,
    policy_type=GeneralDiffVadcPolicy,
)
class GeneralDiffVadc(DiffVadc[GeneralDiffVadcConfig, GeneralDiffVadcPolicy]):
    """Boundary-bucketized voltage ADC with sampling and comparator noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Functional buffers ===

    _boundaries: Tensor  # Shape: [code_num - 1]

    # === Circuit constant buffers ===

    _latency_per_op__ns: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: GeneralDiffVadcConfig,
        policy: GeneralDiffVadcPolicy,
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

        boundaries_t = torch.tensor(config.boundaries, dtype=dtype)
        self.register_buffer("_boundaries", boundaries_t, persistent=False)
        self.register_buffer(
            "_latency_per_op__ns",
            torch.tensor(config.latency_per_op__ns, dtype=dtype),
            persistent=False,
        )

        code_num = boundaries_t.numel() + 1
        self._bits = max(math.ceil(math.log2(code_num)), 1)
        self._code_num = code_num
        self._zero_code = code_num // 2

        if boundaries_t.numel() >= 2:
            self._lsb_estimate = float((boundaries_t[1:] - boundaries_t[:-1]).mean().item())
        else:
            self._lsb_estimate = float(boundaries_t.item())

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def max_bits(self) -> int:
        """Boundary-implied bit width."""
        return self._bits

    @property
    def zero_code(self) -> int:
        """Raw code representing analog zero — the fixed bucket midpoint ``code_num // 2``."""
        return self._zero_code

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Realisable raw code bounds at ``bits`` — ``(0, code_num - 1)``.

        The code count is fixed at construction and may not equal
        ``2 ** bits``. ``bits`` is accepted but does not alter the range.
        """
        del bits
        return 0, self._code_num - 1

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
                Shape: ``[...]``.
            v_neg__V: Negative-side analog input voltage, at the same shape.
                Shape: ``[...]``.
            v_ref__V: Accepted and ignored because the boundaries are fixed.
            bits: Active resolution [bits]; must equal the boundary-implied
                bit width.

        Returns:
            Raw unsigned ``int16`` bucket-index code tensor valued in
            ``[0, code_num - 1]``, at the same shape as ``v_pos__V``.
            Shape: ``[...]``.
        """
        del v_ref__V
        self._validate_runtime_args(bits)
        signal = apply_gaussian(
            v_pos__V - v_neg__V,
            self.config.sampling_noise__V,
            enabled=self.policy.sampling_noise,
        )

        signal = apply_gaussian(
            signal,
            self.config.comparator_noise__V,
            enabled=self.policy.comparator_noise,
        )

        code = floor_bucketize(
            signal,
            self._boundaries,
            out_dtype=torch.int16,
            training=self.training,
            lsb=self._lsb_estimate,
        )

        serial_round_count = self._count_serial_rounds(code.numel())
        latency__ns = self._latency_per_op__ns * serial_round_count
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(code, self.config.energy_per_op__fJ, dtype=torch.float32))
        self._record_latency(latency__ns)

        # Stochastic jitter may cross either outer bucket boundary.
        return code.clamp(min=0, max=self._code_num - 1)

    def _validate_runtime_args(self, bits: int) -> None:
        if bits != self._bits:
            raise ValueError(f"GeneralDiffVadc: bits ({bits}) must equal self._bits ({self._bits})")
