"""Boundary-bucketize voltage ADC with optional Gaussian noise stages.

See Also:
    docs/reference/primitive/analog/diff_voltage_adc/general.md
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import DiffVadc, DiffVadcConfig, DiffVadcPolicy


class GeneralDiffVadcConfig(DiffVadcConfig):
    code_num: int
    """Number of output codes `[0, code_num - 1]`, one more than the
    comparator count."""
    sampling_noise__V: float
    """Input-referred Gaussian sampling-stage noise σ."""
    comparator_noise__V: float
    """Comparator (thermal/decision) noise σ on the signal."""
    energy_per_op__fJ: float
    """Dynamic energy of converting one element."""
    latency_per_op__ns: float
    """Decision window of one conversion, flat across resolutions."""

    def validate(self) -> None:
        super().validate()

        # --- Transfer ---

        self._require_ge(self.code_num, "code_num", 2)

        # --- Noise and PPA ---

        self._require_non_neg(self.sampling_noise__V, "sampling_noise__V")
        self._require_non_neg(self.comparator_noise__V, "comparator_noise__V")
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class GeneralDiffVadcPolicy(DiffVadcPolicy):
    sampling_noise: bool
    """Apply `sampling_noise__V` at convert time."""
    comparator_noise: bool
    """Apply `comparator_noise__V` at convert time."""


@DiffVadc.register_neurox_module(
    config_type=GeneralDiffVadcConfig,
    policy_type=GeneralDiffVadcPolicy,
)
class GeneralDiffVadc(DiffVadc[GeneralDiffVadcConfig, GeneralDiffVadcPolicy]):
    """Boundary-bucketized voltage ADC with sampling and comparator noise.

    A flat comparator bank: `code_num - 1` comparators, whose thresholds arrive
    per call as the injected `v_refs__V` ladder. The ladder is 1-D and
    ascending, because the bucketize runs one shared bank over the whole input.
    """

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

        code_num = config.code_num
        self._bits = max(math.ceil(math.log2(code_num)), 1)
        self._code_num = code_num
        self._tap_num = code_num - 1
        self._zero_code = code_num // 2

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def latency__ns(self, *, bits: int) -> float:
        """One conversion — the flat comparison window.

        The comparator bank fires every ladder tap at once, so the converter
        owns no time axis and the resolution does not lengthen the window.
        """
        del bits
        return self.config.latency_per_op__ns

    @property
    def max_bits(self) -> int:
        """Code-count-implied bit width."""
        return self._bits

    @property
    def zero_code(self) -> int:
        """Raw code representing analog zero — the fixed bucket midpoint `code_num // 2`."""
        return self._zero_code

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Realisable raw code bounds at `bits` — `(0, code_num - 1)`.

        The code count is fixed at construction and may not equal `2 ** bits`;
        `bits` is accepted but does not alter the range.
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
        v_refs__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Quantise a differential analog voltage to a raw code (floor-bucketize).

        Args:
            v_pos__V: Positive-side analog input voltage.
                Shape: `[...]`.
            v_neg__V: Negative-side analog input voltage, at the same shape.
                Shape: `[...]`.
            v_refs__V: Injected comparator thresholds in input units — one
                ascending ladder of `code_num - 1` taps, shared by every input
                position.
                Shape: `[code_num - 1]`.
            bits: Active resolution [bits]; must equal the code-count-implied
                bit width.

        Returns:
            Raw unsigned `int16` bucket-index code tensor valued in
            `[0, code_num - 1]`, one code per `v_pos__V` element.
            Shape: `[...]`.

        Raises:
            ValueError: `bits` differs from the code-count-implied bit width,
                or `v_refs__V` is not a 1-D ladder of `code_num - 1` taps.
        """
        self._validate_runtime_args(v_refs__V, bits)
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

        if self.training:
            jitter = torch.rand(signal.shape, device=signal.device, dtype=signal.dtype) * self._lsb__V(v_refs__V)
            signal = signal + jitter
        # `right=True` gives floor semantics: signal at an exact
        # boundary lands in the upper bin (code = C when signal == C·LSB).
        code = torch.bucketize(signal, v_refs__V, right=True, out_int32=True).to(torch.int16)

        if self._is_dynamic_energy_profile_active():
            # Shape: [] -> [*code.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=code.device)
            e_op__fJ = e_op__fJ.expand(code.shape)
            self._record_dynamic_energy(e_op__fJ)

        # Stochastic jitter may cross either outer bucket boundary.
        return code.clamp(min=0, max=self._code_num - 1)

    def _lsb__V(self, v_refs__V: Tensor) -> Tensor:
        """Mean tap spacing [V], the bin width the stochastic jitter is sized by.

        Args:
            v_refs__V: Injected threshold ladder.
                Shape: `[code_num - 1]`.

        Returns:
            Bin width; the sole tap itself when the bank holds one.
            Shape: `[]`.
        """
        if self._tap_num == 1:
            return v_refs__V[0]
        return (v_refs__V[1:] - v_refs__V[:-1]).mean()

    def _validate_runtime_args(self, v_refs__V: Tensor, bits: int) -> None:
        if bits != self._bits:
            raise ValueError(f"GeneralDiffVadc: bits ({bits}) must equal self._bits ({self._bits})")
        if v_refs__V.ndim != 1:
            raise ValueError(f"require: v_refs__V is a 1-D ladder (ndim {v_refs__V.ndim})")
        tap_num = int(v_refs__V.shape[0])
        if tap_num != self._tap_num:
            raise ValueError(f"require: v_refs__V tap_num ({tap_num}) == code_num - 1 ({self._tap_num})")
