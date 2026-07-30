"""Conv2dUnit operator interface.

See also:
    docs/internals/architecture/unit/conv2d.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .base import UnitBase


class Conv2dUnit(UnitBase, ABC):
    """Interface for an integer ``torch.nn.functional.conv2d`` replacement.

    Grouped convolution is not supported.
    """

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor.
                Shape: ``[C_out, C_in, kh, kw]``.
            bias: Optional integer bias tensor, added in the int64 accumulation
                domain by :meth:`conv2d`; ``None`` clears any programmed bias.
                Shape: ``[C_out]``.
        """
        raise NotImplementedError

    @abstractmethod
    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Convert convolution input to matmul-shaped planes."""
        raise NotImplementedError

    @abstractmethod
    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Fold matmul output back to the convolution output layout.

        Returns:
            Integer convolution output planes.
            Shape: ``[..., C_out, H_out, W_out]``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def conv2d(self, input: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Execute one integer 2-D convolution against the programmed state.

        Args:
            input: Integer activation tensor.
                Shape: ``[..., C_in, H, W]``.
            quantization_mode: Runtime quantization-mode index.
            adc_bits: Runtime ADC resolution, or ``None`` for the lossless
                oracle.

        Returns:
            Integer pre-requantize output tensor; leading dims mirror ``input``.
            Shape: ``[..., C_out, H_out, W_out]``.
        """
        if input.ndim < 3:
            raise ValueError(f"conv2d() expects input with trailing [C_in, H, W]; got ndim {input.ndim}")
        out_hw = self._conv2d_out_hw(input.shape[-2], input.shape[-1])
        planes = self._conv2d_planes(input, out_hw=out_hw)
        y = self._matmul(planes, quantization_mode=quantization_mode, adc_bits=adc_bits)
        y = self._conv2d_fold(y, out_hw=out_hw)
        int_bias = self._int_bias
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
