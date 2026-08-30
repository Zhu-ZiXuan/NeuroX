"""Ideal linear compute unit."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.architecture.unit.linear import LinearUnit


class IdealLinearUnitConfig(CimUnitConfig):
    x_value_range: tuple[int, int]
    """Inclusive integer activation range."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range."""


class IdealLinearUnitPolicy(CimUnitPolicy):
    pass


@CimUnit.register_neurox_module(config_type=IdealLinearUnitConfig, policy_type=IdealLinearUnitPolicy)
class IdealLinearUnit(LinearUnit, CimUnit[IdealLinearUnitConfig, IdealLinearUnitPolicy]):
    """Exact integer linear unit without output quantization.

    Args:
        w_logical_shape: Logical weight shape `(N, K)` bound to `program(...)`.
        dtype: Requested tensor dtype; it does not affect exact integer execution.
        ideal_macro: Accepted without changing this already ideal unit.
    """

    # === Programmed state ===

    _weight: Tensor  # Shape: [N, K]

    def __init__(
        self,
        *,
        config: IdealLinearUnitConfig,
        policy: IdealLinearUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        if len(self._w_logical_shape) != 2:
            raise ValueError(f"w_logical_shape must be (N, K); got {w_logical_shape}")

        # CUDA lacks integer matmul. This bound identifies contractions that
        # IEEE fp32 evaluates exactly while TF32 remains disabled.
        x_lo, x_hi = config.x_value_range
        w_lo, w_hi = config.w_value_range
        max_dot_abs = self._w_logical_shape[-1] * max(abs(x_lo), abs(x_hi)) * max(abs(w_lo), abs(w_hi))
        self._fp32_exact = max_dot_abs < 2**24

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def adc_bits(self) -> int | None:
        return None

    def rescale_factor(self, *, quantization_mode: int, adc_active_bits: int) -> float:
        del quantization_mode, adc_active_bits
        return 1.0

    def initiation_interval__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int) -> float:
        """Zero — an exact integer matmul occupies no execution interval.

        The unit holds neither a macro nor an engine schedule, so there is no
        schedule anywhere below it and the operand layout says nothing about
        an interval.
        """
        del input_shape, adc_active_bits
        return 0.0

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._weight = weight
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])

    @torch.no_grad()
    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        del quantization_mode, adc_active_bits
        weight = self._weight
        if self._fp32_exact:
            # Shape: [..., M, K] @ [K, N] -> [..., M, N]
            out = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
            return out.to(torch.int64)
        # Shape: [..., M, K] @ [K, N] -> [..., M, N]
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
