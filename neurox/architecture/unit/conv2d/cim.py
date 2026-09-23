"""Conv2dCimUnit — CIM convolution with directly owned circuits.

See Also:
    docs/reference/architecture/unit/conv2d.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy

from .base import Conv2dUnit, Conv2dUnitConfig, Conv2dUnitPolicy


class Conv2dCimUnitConfig(Conv2dUnitConfig, CimUnitConfig):
    # === Input placement ===

    merge: bool
    """Whether short input tiles share a macro through separate input slots."""


class Conv2dCimUnitPolicy(Conv2dUnitPolicy, CimUnitPolicy):
    pass


_Config = Conv2dCimUnitConfig
_Policy = Conv2dCimUnitPolicy


@Conv2dUnit.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class Conv2dCimUnit(Conv2dUnit, CimUnit):
    """CIM-backed convolution with independent, parallel hardware per channel group."""

    config: _Config
    policy: _Policy

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
        Conv2dUnit.__init__(
            self,
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
        )
        output_channel, input_channel_per_group, kernel_h, kernel_w = self._w_logical_shape
        CimUnit.__init__(
            self,
            matrix_input_num=input_channel_per_group * kernel_h * kernel_w,
            matrix_output_num=output_channel // groups,
            group_num=groups,
            merge=config.merge,
            dtype=dtype,
        )
        if padding != (0, 0):
            x_lo, x_hi = self.x_slicer.value_range
            if not (x_lo <= 0 <= x_hi):
                raise ValueError(
                    f"require: x_value_range ({(x_lo, x_hi)}) covers 0 — convolution padding injects x = 0"
                )

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int | None) -> float:
        if len(input_shape) < 3 or input_shape[-3] != self._w_logical_shape[1] * self.groups:
            raise ValueError("latency__ns expects [..., input_channel, height, width]")
        if any(size < 0 for size in input_shape):
            raise ValueError("input_shape extents must be nonnegative")
        h_out, w_out = self._conv2d_out_hw(*input_shape[-2:])
        local__ns = self._vmm_local_latency__ns(adc_active_bits=adc_active_bits)
        global__ns = self._vmm_global_latency__ns()
        propagation__ns = self.tiler.input_tile_num * self.config.clock_period__ns
        window_period__ns = max(local__ns, propagation__ns)
        return local__ns + (h_out * w_out - 1) * window_period__ns + global__ns

    @torch.no_grad()
    def program(self, weight: Tensor, *, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        # Shape: [output_channel, input_channel_per_group, kernel_h, kernel_w] -> [output_channel, input]
        matrix = weight.flatten(start_dim=1)
        # Shape: [output_channel, input] -> [group, output, input]
        matrix = matrix.unflatten(0, (self.groups, self.tiler.matrix_output_num))
        self._program_matrix(matrix)
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        out_hw = self._conv2d_out_hw(input.shape[-2], input.shape[-1])
        planes = self._conv2d_planes(input)
        y = self._vmm(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        y = self._conv2d_fold(y, out_hw=out_hw)
        int_bias = self._int_bias
        if int_bias is not None:
            # Shape: [C_out] -> [C_out, H_out=1, W_out=1]
            y = y + int_bias.view(-1, 1, 1)
        return y

    def _conv2d_planes(self, input: Tensor) -> Tensor:
        """Gather convolution windows.

        Returns:
            Flattened input vectors for each convolution window and group.
            Shape: `[*leading, M, group, K]`.
        """
        x = self._conv2d_windows(input)
        # Shape: [..., H_out, W_out, C_in, kh, kw] -> [..., H_out, W_out, groups*K]
        x = x.flatten(start_dim=-3)
        # Shape: [..., H_out, W_out, group*K] -> [..., H_out, W_out, group, K]
        grouped: Tensor = x.unflatten(-1, (self.groups, self.tiler.matrix_input_num))
        # Shape: [..., H_out, W_out, group, K] -> [..., M, group, K]
        return grouped.flatten(start_dim=-4, end_dim=-3)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Restore serial windows to the convolution output layout.

        Returns:
            Folded output map.
            Shape: `[*leading, C_out, H_out, W_out]`.
        """
        h_out, w_out = out_hw
        # Shape: [..., M, group, output] -> [..., M, C_out]
        output = output.flatten(-2)
        # Shape: [..., M, C_out] -> [..., H_out, W_out, C_out]
        y = output.unflatten(-2, (h_out, w_out))
        # Shape: [..., H_out, W_out, C_out] -> [..., C_out, H_out, W_out]
        folded: Tensor = y.movedim(-1, -3)
        return folded
