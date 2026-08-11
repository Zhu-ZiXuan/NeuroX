"""Conv2dCimUnit — engine-backed ``F.conv2d`` replacement.

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
    """CIM-backed convolution using one programmed kernel matrix."""

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
        _c_out, _c_in, kh, kw = w_logical_shape
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
        if config.padding != (0, 0):
            x_lo, x_hi = self.engine.x_value_range
            if not (x_lo <= 0 <= x_hi):
                raise ValueError(
                    f"require: x_value_range ({(x_lo, x_hi)}) covers 0 — convolution padding injects x = 0"
                )

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_bits: int | None) -> float:
        """Time the matmul this convolution lowers to.

        The unit introduces no time axis of its own; it restates the call in
        the engine's terms. ``M = H_out * W_out`` is the one extent no config
        fixes, and it comes from :meth:`_conv2d_out_hw`, the same helper the
        forward gathers windows with.

        Raises:
            ValueError: ``input_shape`` has no ``[C_in, H, W]`` trailing triple.
        """
        if len(input_shape) < 3:
            raise ValueError(f"latency__ns() expects input_shape with trailing [C_in, H, W]; got {input_shape}")
        h, w = input_shape[-2:]
        h_out, w_out = self._conv2d_out_hw(h, w)
        return self.engine.latency__ns(output_plane_num=h_out * w_out, adc_bits=adc_bits)

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Flattened kernel-matrix shape ``(C_out, C_in*kh*kw)``."""
        c_out, c_in, kh, kw = self._w_logical_shape
        return (c_out, c_in * kh * kw)

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Flatten one copy of every output-channel kernel.

        Returns:
            Programmed kernel matrix.
            Shape: ``[C_out, K]``.
        """
        # Shape: [C_out, C_in, kh, kw] -> [C_out, K]
        return weight.flatten(start_dim=1)

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        if weight.dtype.is_floating_point or weight.dtype.is_complex or weight.dtype == torch.bool:
            raise TypeError(f"CIM execution requires an integer weight tensor; got dtype {weight.dtype}")
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Gather convolution windows.

        Returns:
            Matmul-shaped input planes.
            Shape: ``[..., M, K]``.
        """
        h_out, w_out = out_hw
        kh, kw = self._conv2d_kernel_size
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h, d_w = self._conv2d_dilation
        x = input
        if p_h or p_w:
            # Shape: [..., C_in, H, W] -> [..., C_in, Hp, Wp]
            x = F.pad(x, (p_w, p_w, p_h, p_h))
        device = x.device

        # Shape: [H_out, kh]
        h_idx = (torch.arange(h_out, device=device) * s_h).view(-1, 1) + (torch.arange(kh, device=device) * d_h).view(
            1, -1
        )
        # Shape: [W_out, kw]
        w_idx = (torch.arange(w_out, device=device) * s_w).view(-1, 1) + (torch.arange(kw, device=device) * d_w).view(
            1, -1
        )
        # Shape: [..., C_in, Hp, Wp] -> [..., C_in, H_out, kh, Wp]
        x = x[..., h_idx, :]
        # Shape: [..., C_in, H_out, kh, Wp] -> [..., C_in, H_out, kh, W_out, kw]
        x = x[..., w_idx]
        # Shape: [..., C_in, H_out, kh, W_out, kw] -> [..., H_out, W_out, C_in, kh, kw]
        b = x.ndim - 5
        x = x.permute(*range(b), b + 1, b + 3, b + 0, b + 2, b + 4)
        # Shape: [..., H_out, W_out, C_in, kh, kw] -> [..., H_out, W_out, K]
        x = x.flatten(start_dim=-3)
        # Shape: [..., H_out, W_out, K] -> [..., M, K]
        return x.flatten(start_dim=-3, end_dim=-2)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Restore serial windows to the convolution output layout.

        Returns:
            Folded output map.
            Shape: ``[..., C_out, H_out, W_out]``.
        """
        h_out, w_out = out_hw
        # Shape: [..., M, C_out] -> [..., H_out, W_out, C_out]
        y = output.unflatten(-2, (h_out, w_out))
        # Shape: [..., H_out, W_out, C_out] -> [..., C_out, H_out, W_out]
        folded: Tensor = y.movedim(-1, -3)
        return folded
