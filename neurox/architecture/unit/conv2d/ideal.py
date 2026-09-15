"""Ideal conv2d compute unit."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Conv2dUnit, Conv2dUnitConfig, Conv2dUnitPolicy


class IdealConv2dUnitConfig(Conv2dUnitConfig):
    area_per_inst__um2: float
    """Unit-local peripheral area, excluding child circuits."""
    leakage_per_inst__uW: float
    """Unit-local static leakage, excluding child circuits."""

    x_value_range: tuple[int, int]
    """Inclusive integer activation range."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IdealConv2dUnitPolicy(Conv2dUnitPolicy):
    pass


_Config = IdealConv2dUnitConfig
_Policy = IdealConv2dUnitPolicy


@Conv2dUnit.register_impl(config_type=_Config, policy_type=_Policy)
class IdealConv2dUnit(Conv2dUnit):
    """Integer convolution evaluation through `torch.nn.functional.conv2d`.

    Args:
        w_logical_shape: Logical kernel shape `(C_out, C_in, kh, kw)` bound to `program(...)`.
    """

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _weight: Tensor  # Shape: [output_channel, input_channel, kernel_h, kernel_w]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
        )

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
        """Zero — an exact integer convolution has no modeled latency.

        The unit has no modeled circuit latency; all output
        positions are evaluated at once.
        """
        del input_shape, adc_active_bits
        return 0.0

    @torch.no_grad()
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._weight = weight.long()
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        return F.conv2d(
            input.long(),
            self._weight,
            self._int_bias,
            stride=self.config.stride,
            padding=self.config.padding,
            dilation=self.config.dilation,
        )
