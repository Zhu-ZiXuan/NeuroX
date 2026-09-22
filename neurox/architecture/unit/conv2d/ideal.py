"""Ideal conv2d compute unit."""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    """Integer convolution evaluation through `torch.nn.functional.conv2d`.

    Args:
        w_logical_shape: Logical kernel shape `(C_out, C_in/groups, kh, kw)` bound to `program(...)`.
    """

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
        dtype: torch.dtype,
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
        adc_active_bits: int | None,
    ) -> float:
        del quantization_mode, adc_active_bits
        return 1.0

    def latency__ns(
        self,
        input_shape: tuple[int, ...],
        *,
        adc_active_bits: int | None,
    ) -> float:
        """Zero — one image's ideal convolution has no modeled latency.

        The operand layout does not change the zero modeled duration.
        """
        return 0.0

    @torch.no_grad()
    def program(
        self,
        weight: Tensor,
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
        return F.conv2d(
            input.long(),
            self._weight,
            self._int_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
