"""Ideal conv2d compute unit."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Conv2dUnit, Conv2dUnitConfig, Conv2dUnitPolicy


class IdealConv2dUnitConfig(Conv2dUnitConfig):
    # === Value ranges ===

    x_value_range: tuple[int, int]
    """Inclusive integer activation range."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range."""


class IdealConv2dUnitPolicy(Conv2dUnitPolicy):
    pass


_Config = IdealConv2dUnitConfig
_Policy = IdealConv2dUnitPolicy


@Conv2dUnit.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class IdealConv2dUnit(Conv2dUnit):
    """Exact int64 convolution on the programmed device."""

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _weight: Tensor  # Shape: [output_channel, input_channel_per_group, kernel_h, kernel_w]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
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
        """Zero — one image's ideal convolution has no modeled latency.

        The operand layout does not change the zero modeled duration.
        """
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
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_impl(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        self._conv2d_out_hw(input.shape[-2], input.shape[-1])
        # CUDA convolution backends do not support int64. Keep the reduction on device.
        windows = self._conv2d_windows(input.long())
        # Shape: [..., H_out, W_out, C_in, kh, kw] -> [..., H_out, W_out, group, K]
        windows = windows.unflatten(-3, (self.groups, self._w_logical_shape[1])).flatten(-3)
        # Shape: [C_out, C_in/group, kh, kw] -> [group, C_out/group, K]
        weight = self._weight.flatten(1).unflatten(0, (self.groups, self._w_logical_shape[0] // self.groups))
        # Shape: [..., H_out, W_out, group, C_out/group, K] -> [..., C_out, H_out, W_out]
        output: Tensor = (windows.unsqueeze(-2) * weight).sum(dim=-1, dtype=torch.int64).flatten(-2).movedim(-1, -3)
        if self._int_bias is not None:
            output = output + self._int_bias.view(-1, 1, 1)
        return output
