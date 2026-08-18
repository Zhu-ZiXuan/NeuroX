"""Ideal conv2d compute unit."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.architecture.unit.conv2d import Conv2dUnit


class IdealConv2dUnitConfig(CimUnitConfig):
    x_value_range: tuple[int, int]
    """Inclusive integer activation range."""
    w_value_range: tuple[int, int]
    """Inclusive integer weight range."""
    stride: tuple[int, int]
    """Output step `(s_h, s_w)`."""
    padding: tuple[int, int]
    """Zero-pad extent `(p_h, p_w)` on each side."""
    dilation: tuple[int, int]
    """Kernel tap spacing `(d_h, d_w)`."""

    def validate(self) -> None:
        super().validate()

        self._require_pos(self.stride[0], "stride[0]")
        self._require_pos(self.stride[1], "stride[1]")
        self._require_pos(self.dilation[0], "dilation[0]")
        self._require_pos(self.dilation[1], "dilation[1]")
        self._require_non_neg(self.padding[0], "padding[0]")
        self._require_non_neg(self.padding[1], "padding[1]")


class IdealConv2dUnitPolicy(CimUnitPolicy):
    pass


@CimUnit.register_neurox_module(config_type=IdealConv2dUnitConfig, policy_type=IdealConv2dUnitPolicy)
class IdealConv2dUnit(Conv2dUnit, CimUnit[IdealConv2dUnitConfig, IdealConv2dUnitPolicy]):
    """Exact integer convolution unit without output quantization.

    Args:
        w_logical_shape: Logical kernel shape `(C_out, C_in, kh, kw)` bound to `program(...)`.
        dtype: Requested tensor dtype; it does not affect exact integer execution.
        ideal_macro: Accepted without changing this already ideal unit.
    """

    # === Programmed state ===

    _weight: Tensor  # Shape: [C_out, C_in*kh*kw]

    def __init__(
        self,
        *,
        config: IdealConv2dUnitConfig,
        policy: IdealConv2dUnitPolicy,
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
        if len(self._w_logical_shape) != 4:
            raise ValueError(f"w_logical_shape must be (C_out, C_in, kh, kw); got {w_logical_shape}")
        self._init_conv2d_operator(
            kernel_size=(self._w_logical_shape[-2], self._w_logical_shape[-1]),
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
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
    def adc_max_bits(self) -> int | None:
        return None

    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        del quantization_mode, adc_bits
        return 1.0

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_bits: int | None) -> float:
        """Zero — an exact integer matmul, with no circuit under it to take time.

        The unit holds neither a macro nor an engine schedule, so there is no
        time axis anywhere below it: the output positions the input resolution
        implies are all evaluated at once.
        """
        del input_shape, adc_bits
        return 0.0

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        # Shape: [C_out, C_in, kh, kw] -> [C_out, C_in*kh*kw]
        return weight.flatten(start_dim=1).to(torch.int64)

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self._weight = self._weight_to_matrix(weight.detach())
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        h_out, w_out = out_hw
        kh, kw = self._conv2d_kernel_size
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h, d_w = self._conv2d_dilation

        x = input
        if p_h > 0 or p_w > 0:
            # Shape: [B, C_in, H, W] -> [B, C_in, Hp, Wp]
            x = F.pad(x, (p_w, p_w, p_h, p_h))

        device = x.device
        # Index-grid gather rather than `F.unfold`: pure data movement, so the
        # oracle's window planes stay exact in int64.
        rows = (torch.arange(h_out, device=device) * s_h).view(h_out, 1, 1, 1) + (
            torch.arange(kh, device=device) * d_h
        ).view(1, 1, kh, 1)
        cols = (torch.arange(w_out, device=device) * s_w).view(1, w_out, 1, 1) + (
            torch.arange(kw, device=device) * d_w
        ).view(1, 1, 1, kw)

        # Shape: [B, C_in, Hp, Wp] -> [B, C_in, H_out, W_out, kh, kw]
        patches = x[..., rows, cols]
        # Shape: [B, C_in, H_out, W_out, kh, kw] -> [B, H_out, W_out, C_in, kh, kw]
        patches = patches.movedim(-5, -3)
        # Shape: [B, H_out, W_out, C_in, kh, kw] -> [B, L, C_in*kh*kw]
        return patches.flatten(-3).flatten(-3, -2)

    @torch.no_grad()
    def _matmul(self, planes: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        del quantization_mode, adc_bits
        # Shape: [B, L, C_in*kh*kw] @ [C_in*kh*kw, C_out] -> [B, L, C_out]
        return planes.to(torch.int64) @ self._weight.transpose(-2, -1)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        # Shape: [B, L, C_out] -> [B, C_out, H_out, W_out]
        return output.transpose(-2, -1).unflatten(-1, out_hw)
