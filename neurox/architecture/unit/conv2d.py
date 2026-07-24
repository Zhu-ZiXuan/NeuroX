"""Conv2dUnit operator ABC and the ideal conv2d reference unit.

See also:
    docs/internals/architecture/unit/conv2d.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.base import UnitBase


class Conv2dUnit(UnitBase, ABC):
    """Interface for an integer ``torch.nn.functional.conv2d`` replacement.

    Grouped convolution is not supported.
    """

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor of shape ``[C_out, C_in, kh, kw]``.
            bias: Optional integer bias tensor of shape ``(C_out,)``, added
                in the int64 accumulation domain by :meth:`conv2d`;
                ``None`` clears any programmed bias.
        """
        raise NotImplementedError

    @abstractmethod
    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Convert convolution input to matmul-shaped planes."""
        raise NotImplementedError

    @abstractmethod
    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Fold matmul output to ``[..., C_out, H_out, W_out]``."""
        raise NotImplementedError

    @torch.no_grad()
    def conv2d(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Execute one integer 2-D convolution against the programmed state.

        Args:
            input: Integer activation tensor with trailing ``[C_in, H, W]``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC resolution.

        Returns:
            Integer pre-requantize output tensor with trailing
            ``[C_out, H_out, W_out]``; leading dims mirror ``input``.
        """
        if input.ndim < 3:
            raise ValueError(f"conv2d() expects input with trailing [C_in, H, W]; got ndim {input.ndim}")
        out_hw = self._conv2d_out_hw(input.shape[-2], input.shape[-1])
        planes = self._conv2d_planes(input, out_hw=out_hw)
        y = self._matmul(planes, adc_mode=adc_mode, adc_bits=adc_bits)
        y = self._conv2d_fold(y, out_hw=out_hw)
        int_bias = self.int_bias
        if int_bias is not None:
            # Shape: [C_out] -> [C_out, 1, 1]
            y = y + int_bias.view(-1, 1, 1)
        return y

    def _conv2d_out_hw(self, h: int, w: int) -> tuple[int, int]:
        """Output map extent ``(H_out, W_out)`` for an ``(h, w)`` input map.

        Raises:
            ValueError: the configured geometry yields an empty output map.
        """
        kh, kw = self._conv2d_kernel_size
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h, d_w = self._conv2d_dilation
        h_out = (h + 2 * p_h - d_h * (kh - 1) - 1) // s_h + 1
        w_out = (w + 2 * p_w - d_w * (kw - 1) - 1) // s_w + 1
        if h_out < 1 or w_out < 1:
            raise ValueError(f"require: positive output map; got (H_out, W_out) = {(h_out, w_out)}")
        return (h_out, w_out)

    def _init_conv2d_operator(
        self,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        """Store the convolution geometry.

        Args:
            kernel_size: Kernel map extent ``(kh, kw)``.
            stride: Output step ``(s_h, s_w)``.
            padding: Zero-pad extent ``(p_h, p_w)`` on each side.
            dilation: Kernel tap spacing ``(d_h, d_w)``.
        """
        self._conv2d_kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self._conv2d_stride = (int(stride[0]), int(stride[1]))
        self._conv2d_padding = (int(padding[0]), int(padding[1]))
        self._conv2d_dilation = (int(dilation[0]), int(dilation[1]))


# Deferred to avoid a circular import with CIM unit implementations.
from neurox.architecture.unit.cim.base import CimUnit, CimUnitConfig, CimUnitPolicy  # noqa: E402


class IdealConv2dUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealConv2dUnit`.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
        stride: Output step ``(s_h, s_w)``.
        padding: Zero-pad extent ``(p_h, p_w)`` on each side.
        dilation: Kernel tap spacing ``(d_h, d_w)``.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def validate(self) -> None:
        super().validate()
        self.validate_geometry()

    def validate_geometry(self) -> None:
        """Require positive stride / dilation and non-negative padding."""
        self._require_pos(self.stride[0], "stride[0]")
        self._require_pos(self.stride[1], "stride[1]")
        self._require_pos(self.dilation[0], "dilation[0]")
        self._require_pos(self.dilation[1], "dilation[1]")
        self._require_non_neg(self.padding[0], "padding[0]")
        self._require_non_neg(self.padding[1], "padding[1]")


class IdealConv2dUnitPolicy(CimUnitPolicy):
    """Empty policy — :class:`IdealConv2dUnit` has no nonidealities to toggle."""


@CimUnit.register_key(IdealConv2dUnitConfig)
class IdealConv2dUnit(Conv2dUnit, CimUnit[IdealConv2dUnitConfig, IdealConv2dUnitPolicy]):
    """Exact integer convolution unit without output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Nonideality policy.
        w_logical_shape: Logical kernel shape ``(C_out, C_in, kh, kw)`` bound to ``program(...)``.
        dtype: Requested tensor dtype; it does not affect exact integer execution.
        T__K: Operating temperature.
        ideal_xbar: Accepted without changing this already ideal unit.
    """

    def __init__(
        self,
        *,
        config: IdealConv2dUnitConfig,
        policy: IdealConv2dUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        if len(self._w_logical_shape) != 4:
            raise ValueError(f"w_logical_shape must be (C_out, C_in, kh, kw); got {w_logical_shape}")
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._init_conv2d_operator(
            kernel_size=(self._w_logical_shape[-2], self._w_logical_shape[-1]),
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def adc_mode_num(self) -> int:
        return 1

    @property
    def adc_max_bits(self) -> int:
        return 0

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        del adc_mode, adc_bits
        return 1.0

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        # Shape: [C_out, C_in, kh, kw] -> [C_out, C_in*kh*kw]
        return weight.flatten(start_dim=1).to(torch.int64)

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        if weight.is_floating_point() or weight.is_complex():
            raise TypeError(f"program() expects an integer weight tensor; got dtype {weight.dtype}")
        self.weight = self._weight_to_matrix(weight.detach())
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        h_out, w_out = out_hw
        kh, kw = self._conv2d_kernel_size
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h, d_w = self._conv2d_dilation

        x = input
        if p_h > 0 or p_w > 0:
            # Shape: [..., C_in, H, W] -> [..., C_in, Hp, Wp]
            x = F.pad(x, (p_w, p_w, p_h, p_h))

        device = x.device
        rows = (torch.arange(h_out, device=device) * s_h).view(h_out, 1, 1, 1) + (
            torch.arange(kh, device=device) * d_h
        ).view(1, 1, kh, 1)
        cols = (torch.arange(w_out, device=device) * s_w).view(1, w_out, 1, 1) + (
            torch.arange(kw, device=device) * d_w
        ).view(1, 1, 1, kw)

        # Shape: [..., C_in, Hp, Wp] -> [..., C_in, H_out, W_out, kh, kw]
        patches = x[..., rows, cols]
        # Shape: [..., C_in, H_out, W_out, kh, kw] -> [..., H_out, W_out, C_in, kh, kw]
        patches = patches.movedim(-5, -3)
        # Shape: [..., H_out, W_out, C_in, kh, kw] -> [..., L, C_in*kh*kw]
        return patches.flatten(-3).flatten(-3, -2)

    @torch.no_grad()
    def _matmul(self, planes: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        del adc_mode, adc_bits
        # Shape: [..., L, C_in*kh*kw] @ [C_in*kh*kw, C_out] -> [..., L, C_out]
        return planes.to(torch.int64) @ self.weight.transpose(-2, -1)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        # Shape: [..., L, C_out] -> [..., C_out, H_out, W_out]
        return output.transpose(-2, -1).unflatten(-1, out_hw)
