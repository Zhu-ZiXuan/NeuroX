"""Ideal linear compute unit."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.cim.base import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.architecture.unit.linear import LinearUnit


class IdealLinearUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealLinearUnit`.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


class IdealLinearUnitPolicy(CimUnitPolicy):
    """Policy for :class:`IdealLinearUnit`."""


@CimUnit.register_neurox_module(config_type=IdealLinearUnitConfig, policy_type=IdealLinearUnitPolicy)
class IdealLinearUnit(LinearUnit, CimUnit[IdealLinearUnitConfig, IdealLinearUnitPolicy]):
    """Exact integer linear unit without output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Runtime policy.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Requested tensor dtype; it does not affect exact integer execution.
        T__K: Operating temperature.
        ideal_macro: Accepted without changing this already ideal unit.
    """

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
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

        # CUDA lacks integer matmul. This bound identifies contractions that
        # IEEE fp32 evaluates exactly while TF32 remains disabled.
        x_lo, x_hi = config.x_value_range
        w_lo, w_hi = config.w_value_range
        max_dot_abs = self._w_logical_shape[-1] * max(abs(x_lo), abs(x_hi)) * max(abs(w_lo), abs(w_hi))
        self._fp32_exact = max_dot_abs < 2**24

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def adc_max_bits(self) -> int | None:
        return None

    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        del quantization_mode, adc_bits
        return 1.0

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._weight = weight
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])

    @torch.no_grad()
    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        del quantization_mode, adc_bits
        weight = self._weight
        if self._fp32_exact:
            # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
            out = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
            return out.to(torch.int64)
        # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
