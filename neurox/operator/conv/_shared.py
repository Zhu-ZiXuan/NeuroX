"""Shared im2col / fold plumbing for the Conv2d operators.

Both :class:`QuantConv2d` and :class:`HATConv2d` regroup their input
tensor into ``[batch_flat, groups, L, K]`` form before delegating to
the same macro matmul kernel that the Linear operators use.  These
helpers package the unfold / reshape / fold geometry so the two
operators share one source of truth — any conv-layout fix lands in
both at once.

The functions are kept private (``_``-prefixed) since they're not
intended as a public API; they live here only to be re-used between
``quant_conv2d.py`` and ``hat_conv2d.py``.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple


def _build_reversed_padding(
    padding: str | tuple[int, int],
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
) -> list[int]:
    """Build padding metadata matching PyTorch Conv2d internals."""
    if isinstance(padding, str):
        if padding == "valid":
            return [0, 0, 0, 0]
        if padding != "same":
            raise ValueError(f"Unsupported padding string: {padding}")
        reversed_padding = [0, 0] * len(kernel_size)
        for axis in range(len(kernel_size) - 1, -1, -1):
            total_padding = dilation[axis] * (kernel_size[axis] - 1)
            left_padding = total_padding // 2
            index = 2 * (len(kernel_size) - 1 - axis)
            reversed_padding[index] = left_padding
            reversed_padding[index + 1] = total_padding - left_padding
        return reversed_padding
    return list(_reverse_repeat_tuple(padding, 2))


def _unfold_input(
    input: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    padding_mode: str,
    reversed_padding_repeated_twice: list[int],
) -> tuple[Tensor, tuple[int, ...], int, int]:
    """Convert input tensor to grouped im2col form.

    Returns ``(unfolded, batch_shape, out_h, out_w)`` where ``unfolded``
    has shape ``[batch_flat, groups, L, K]`` with ``L = out_h * out_w``.
    """
    if padding_mode != "zeros" or isinstance(padding, str):
        padded = F.pad(input, reversed_padding_repeated_twice, mode=padding_mode)
        pad_h, pad_w = 0, 0
    else:
        padded = input
        pad_h, pad_w = _pair(padding)

    batch_shape = padded.shape[:-3]
    in_c, in_h, in_w = padded.shape[-3:]
    input_4d = padded.reshape(-1, in_c, in_h, in_w)

    cols = F.unfold(
        input_4d,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=(pad_h, pad_w),
        stride=stride,
    )

    stride_h, stride_w = _pair(stride)
    dilation_h, dilation_w = _pair(dilation)
    kernel_h, kernel_w = _pair(kernel_size)
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    batch_size = input_4d.shape[0]
    k_per_group = cols.shape[1] // groups
    unfolded = cols.view(batch_size, groups, k_per_group, -1).transpose(-2, -1)

    return unfolded, batch_shape, out_h, out_w


def _fold_output(
    output: Tensor,
    out_channels: int,
    batch_shape: tuple[int, ...],
    out_h: int,
    out_w: int,
) -> Tensor:
    """Fold macro output ``[batch_flat, groups, L, out_per_group]`` back to ``[..., C_out, H, W]``."""
    batch_size = output.shape[0]
    output = output.transpose(1, 2)
    output = output.reshape(batch_size, -1, out_channels).transpose(1, 2)
    return output.view(*batch_shape, out_channels, out_h, out_w)


def _reshape_weight(weight: Tensor, groups: int) -> Tensor:
    """Reshape ``[C_out, C_in/groups, kH, kW]`` → ``[groups, out_per_group, K]``."""
    out_channels = weight.shape[0]
    out_per_group = out_channels // groups
    return weight.view(groups, out_per_group, -1)


def _conv_padding_args(
    module: nn.Conv2d,
) -> tuple[tuple[int, int], tuple[int, int], str | tuple[int, int], tuple[int, int]]:
    """Extract normalized conv params from ``nn.Conv2d``."""
    return (
        (module.kernel_size[0], module.kernel_size[1]),
        (module.stride[0], module.stride[1]),
        module.padding if isinstance(module.padding, str) else (module.padding[0], module.padding[1]),
        (module.dilation[0], module.dilation[1]),
    )
