"""Ideal linear compute unit."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import LinearUnit, LinearUnitConfig, LinearUnitPolicy


class IdealLinearUnitConfig(LinearUnitConfig):
    # === Value ranges ===

    x_value_range: tuple[int, int]
    """Inclusive integer activation range."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range."""


class IdealLinearUnitPolicy(LinearUnitPolicy):
    pass


_Config = IdealLinearUnitConfig
_Policy = IdealLinearUnitPolicy


@LinearUnit.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class IdealLinearUnit(LinearUnit):
    """Exact int64 linear evaluation on the programmed device."""

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _weight: Tensor  # Shape: [N, K]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
        )

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def adc_bits(self) -> int | None:
        return None

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None = None,
    ) -> float:
        return 1.0

    def latency__ns(
        self,
        input_shape: tuple[int, ...],
        *,
        adc_active_bits: int | None = None,
    ) -> float:
        return 0.0

    @torch.no_grad()
    def program(
        self,
        weight: Tensor,
        *,
        bias: Tensor | None = None,
    ) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._weight = weight.long()
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])

    def _linear_impl(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        output = (input.long().unsqueeze(-2) * self._weight).sum(dim=-1, dtype=torch.int64)
        return output if self._int_bias is None else output + self._int_bias
