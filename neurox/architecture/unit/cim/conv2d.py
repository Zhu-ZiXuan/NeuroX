"""Conv2dCimUnit — engine-backed ``F.conv2d`` replacement with a Toeplitz weight mapping.

See also:
    docs/internals/architecture/unit/cim/conv2d.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.conv2d import Conv2dUnit

from .base import (
    CimUnit,
    EngineBackedCimUnit,
    EngineBackedCimUnitConfig,
    EngineBackedCimUnitPolicy,
)


class Conv2dCimUnitConfig(EngineBackedCimUnitConfig):
    """Configuration for :class:`Conv2dCimUnit`.

    Attributes:
        stride: Output step ``(s_h, s_w)``.
        padding: Zero-pad extent ``(p_h, p_w)`` on each side.
        dilation: Kernel tap spacing ``(d_h, d_w)``.
    """

    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def validate(self) -> None:
        super().validate()

        self._require_pos(self.stride[0], "stride[0]")
        self._require_pos(self.stride[1], "stride[1]")
        self._require_pos(self.dilation[0], "dilation[0]")
        self._require_pos(self.dilation[1], "dilation[1]")
        self._require_non_neg(self.padding[0], "padding[0]")
        self._require_non_neg(self.padding[1], "padding[1]")


class Conv2dCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for :class:`Conv2dCimUnit`; no fields beyond the inherited set."""


@CimUnit.register_neurox_module(config_type=Conv2dCimUnitConfig, policy_type=Conv2dCimUnitPolicy)
class Conv2dCimUnit(Conv2dUnit, EngineBackedCimUnit[Conv2dCimUnitConfig, Conv2dCimUnitPolicy]):
    """CIM-backed convolution using a Toeplitz input-stationary mapping."""

    def __init__(
        self,
        *,
        config: Conv2dCimUnitConfig,
        policy: Conv2dCimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_macro: bool,
    ) -> None:
        if len(w_logical_shape) != 4:
            raise ValueError(f"w_logical_shape must be (C_out, C_in, kh, kw); got {w_logical_shape}")
        c_out, c_in, kh, kw = w_logical_shape
        self._init_toeplitz_geometry(
            config=config,
            c_out=c_out,
            c_in=c_in,
            kh=kh,
            kw=kw,
        )
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_macro=ideal_macro,
        )
        self._init_conv2d_operator(
            kernel_size=(kh, kw),
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )
        w_lo, w_hi = self.engine.w_value_range
        if not (w_lo <= 0 <= w_hi):
            raise ValueError(
                f"require: w_value_range ({(w_lo, w_hi)}) covers 0 — the Toeplitz matrix stores structural zeros"
            )
        if config.padding != (0, 0) or self._w_g > 1:
            x_lo, x_hi = self.engine.x_value_range
            if not (x_lo <= 0 <= x_hi):
                raise ValueError(
                    f"require: x_value_range ({(x_lo, x_hi)}) covers 0 — zero-padding, strip "
                    "right-padding, and last-segment surplus windows inject x = 0 activations"
                )

    def _init_toeplitz_geometry(
        self,
        *,
        config: Conv2dCimUnitConfig,
        c_out: int,
        c_in: int,
        kh: int,
        kw: int,
    ) -> None:
        """Derive the matrix geometry passed to the execution engine."""
        input_num = config.engine.input_num
        output_num = config.engine.output_num
        s_w = config.stride[1]
        d_w = config.dilation[1]
        self._kw_eff = (kw - 1) * d_w + 1
        # Choose the largest window group that fits both logical macro ports.
        g_k = 1 + (input_num // (c_in * kh) - self._kw_eff) // s_w
        g_n = output_num // c_out
        self._w_g = max(1, min(g_k, g_n))
        self._w_strip = self._kw_eff + (self._w_g - 1) * s_w
        self._k_prime = c_in * kh * self._w_strip
        self._n_prime = self._w_g * c_out

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Toeplitz matrix shape ``(N', K')`` handed to the engine."""
        return (self._n_prime, self._k_prime)

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Build the Toeplitz weight matrix with shape ``[N', K']``."""
        c_out, c_in, kh, kw = weight.shape
        w_g, w_strip = self._w_g, self._w_strip
        s_w = self._conv2d_stride[1]
        d_w = self._conv2d_dilation[1]
        device = weight.device
        g = torch.arange(w_g, device=device).view(-1, 1, 1, 1, 1)
        n = torch.arange(c_out, device=device).view(1, -1, 1, 1, 1)
        ci = torch.arange(c_in, device=device).view(1, 1, -1, 1, 1)
        i = torch.arange(kh, device=device).view(1, 1, 1, -1, 1)
        j = torch.arange(kw, device=device).view(1, 1, 1, 1, -1)
        # Shape: [W_g, C_out, C_in, kh, kw]
        c_idx = (g * c_out + n).expand(w_g, c_out, c_in, kh, kw)
        r_idx = ((ci * kh + i) * w_strip + g * s_w + j * d_w).expand(w_g, c_out, c_in, kh, kw)
        matrix = weight.new_zeros(w_g * c_out, c_in * kh * w_strip)
        # Shape: [C_out, C_in, kh, kw] -> [W_g, C_out, C_in, kh, kw] -> [N_prime, K_prime]
        matrix[c_idx.flatten(), r_idx.flatten()] = weight.unsqueeze(0).expand(w_g, -1, -1, -1, -1).flatten()
        return matrix

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        if weight.dtype.is_floating_point or weight.dtype.is_complex or weight.dtype == torch.bool:
            raise TypeError(f"CIM execution requires an integer weight tensor; got dtype {weight.dtype}")
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Gather input strips with shape ``[..., H_out, T_seg, K']``."""
        h_out, w_out = out_hw
        kh = self._conv2d_kernel_size[0]
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h = self._conv2d_dilation[0]
        w_g, w_strip = self._w_g, self._w_strip
        t_seg = -(-w_out // w_g)
        h_in, w_in = input.shape[-2:]
        # Cover the final strip's surplus windows with zero padding.
        pad_bottom = max(0, (h_out - 1) * s_h + (kh - 1) * d_h + 1 - p_h - h_in)
        pad_right = max(0, (t_seg - 1) * w_g * s_w + w_strip - p_w - w_in)
        x = input
        if p_h or p_w or pad_bottom or pad_right:
            # Shape: [..., C_in, H, W] -> [..., C_in, Hp, Wp]
            x = F.pad(x, (p_w, pad_right, p_h, pad_bottom))
        device = x.device
        # Shape: [H_out, kh]
        h_idx = (torch.arange(h_out, device=device) * s_h).view(-1, 1) + (torch.arange(kh, device=device) * d_h).view(
            1, -1
        )
        # Shape: [T_seg, W_strip]
        w_idx = (torch.arange(t_seg, device=device) * (w_g * s_w)).view(-1, 1) + torch.arange(
            w_strip, device=device
        ).view(1, -1)
        # Shape: [..., C_in, Hp, Wp] -> [..., C_in, H_out, kh, Wp]
        x = x[..., h_idx, :]
        # Shape: [..., C_in, H_out, kh, Wp] -> [..., C_in, H_out, kh, T_seg, W_strip]
        x = x[..., w_idx]
        # Shape: [..., C_in, H_out, kh, T_seg, W_strip] -> [..., H_out, T_seg, C_in, kh, W_strip]
        b = x.ndim - 5
        x = x.permute(*range(b), b + 1, b + 3, b + 0, b + 2, b + 4)
        # Shape: [..., H_out, T_seg, C_in, kh, W_strip] -> [..., H_out, T_seg, K']
        return x.flatten(-3)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Fold serial axes to ``[..., C_out, H_out, W_out]``."""
        _h_out, w_out = out_hw
        # Shape: [..., H_out, T_seg, W_g*C_out] -> [..., H_out, T_seg, W_g, C_out]
        y = output.unflatten(-1, (self._w_g, self._w_logical_shape[0]))
        # Shape: [..., H_out, T_seg, W_g, C_out] -> [..., H_out, T_seg*W_g, C_out]
        y = y.flatten(-3, -2)
        # Shape: [..., H_out, T_seg*W_g, C_out] -> [..., H_out, W_out, C_out]
        y = y[..., :w_out, :]
        # Shape: [..., H_out, W_out, C_out] -> [..., C_out, H_out, W_out]
        return y.movedim(-1, -3)
