"""Conv2dUnit operator interface.

See Also:
    docs/reference/architecture/unit/conv2d.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.base import UnitBase, UnitConfig
from neurox.common.module import PolicyBase
from neurox.common.registry_mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealConv2dUnit


class Conv2dUnitConfig(UnitConfig, base_only=True):
    pass


class Conv2dUnitPolicy(PolicyBase, base_only=True):
    pass


_Config = Conv2dUnitConfig
_Policy = Conv2dUnitPolicy


class Conv2dUnit(RegistryMixin[_Config, _Policy], UnitBase, ABC, base_only=True):
    """Interface for an integer `torch.nn.functional.conv2d` replacement.

    Construction initializes the common unit and retains logical kernel shape,
    dtype, and convolution geometry for `to_ideal`. Initialize any additional
    implementation base explicitly after this constructor returns.

    `w_logical_shape` accepts `weight.shape` and is stored as a fixed-length
    `(output_channel, input_channel_per_group, kernel_h, kernel_w)` tuple.
    Output channels must be divisible by `groups`. Runtime inputs have
    `input_channel_per_group * groups` channels. Invalid shapes raise
    `ValueError`.
    One basic operation for latency and profiling is one complete image,
    including its internal window schedule.

    Args:
        stride: Positive output step `(s_h, s_w)`.
        padding: Nonnegative zero-pad extent `(p_h, p_w)` on each side.
        dilation: Positive kernel tap spacing `(d_h, d_w)`.
        groups: Positive number of independent channel groups.
    """

    config: _Config
    policy: _Policy

    # === Programmed state ===

    _int_bias: Tensor | None = None  # Shape: [output_channel]

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
        if groups < 1:
            raise ValueError(f"groups must be positive; got {groups}")
        if len(stride) != 2 or any(step < 1 for step in stride):
            raise ValueError(f"stride must contain two positive steps; got {stride}")
        if len(padding) != 2 or any(extent < 0 for extent in padding):
            raise ValueError(f"padding must contain two nonnegative extents; got {padding}")
        if len(dilation) != 2 or any(step < 1 for step in dilation):
            raise ValueError(f"dilation must contain two positive spacings; got {dilation}")
        if len(w_logical_shape) != 4:
            raise ValueError(f"conv2d w_logical_shape must have 4 axes; got {w_logical_shape}")
        if any(size <= 0 for size in w_logical_shape):
            raise ValueError(f"conv2d weight dimensions must be positive; got {w_logical_shape}")
        if w_logical_shape[0] % groups != 0:
            raise ValueError("conv2d output channels must be divisible by groups")
        UnitBase.__init__(self, config=config, policy=policy)
        self._w_logical_shape = w_logical_shape
        self._dtype = dtype
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        dtype: torch.dtype,
    ) -> Conv2dUnit:
        """Construct the conv2d implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
        )

    @final
    def to_ideal(self) -> IdealConv2dUnit:
        """Construct a fresh ideal conv2d unit with the same logical parameters.

        Preserve value ranges, weight shape, dtype, temperature and unit-local
        static PPA, plus convolution geometry. Weights, bias and child circuits
        are not copied.
        """
        from .ideal import IdealConv2dUnit, IdealConv2dUnitConfig, IdealConv2dUnitPolicy

        config = IdealConv2dUnitConfig(
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            area_per_inst__um2=self._area_per_inst__um2,
            leakage_per_inst__uW=self._leakage_per_inst__uW,
        )
        ideal = IdealConv2dUnit(
            config=config,
            policy=IdealConv2dUnitPolicy(),
            w_logical_shape=self._w_logical_shape,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            dtype=self._dtype,
        )
        ideal.set_temperature(self.T__K)
        return ideal

    @final
    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=True)
    def conv2d(
        self,
        input: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        """Execute one integer 2-D convolution against the programmed state.

        A 3-D `[C_in, H, W]` input is treated as `B = 1` and returns a 3-D
        output, exactly as `torch.nn.functional.conv2d`.

        Args:
            input: Integer activation values.
                Shape: `[B, C_in, H, W]`.
            quantization_mode: Index selecting the runtime quantization window.
            adc_active_bits: Active ADC resolution; `None` requests the
                unit's highest available precision.

        Returns:
            Integer pre-requantize output tensor.
            Shape: `[B, C_out, H_out, W_out]`.

        Raises:
            ValueError: `input` is neither `[B, C_in, H, W]` nor its unbatched
                `[C_in, H, W]` form, its channel count does not match the
                grouped weights, or the configured geometry yields an empty output map.
        """
        if input.ndim not in (3, 4):
            raise ValueError(f"conv2d() expects input [C_in, H, W] or [B, C_in, H, W]; got ndim {input.ndim}")
        input_channels = self._w_logical_shape[1] * self.groups
        if input.shape[-3] != input_channels:
            raise ValueError(f"conv2d() expects {input_channels} input channels; got {input.shape[-3]}")
        output = self._conv2d_impl(input, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        if self._is_profiler_active():
            latency__ns = self.latency__ns(input.shape, adc_active_bits=adc_active_bits)
            sample_shape = (input.shape[0],) if input.ndim == 4 else (1,)
            latency = torch.tensor(latency__ns, dtype=torch.float64)
            self._record_latency(latency.expand(sample_shape[: self._profile_leading_rank]))
        return output

    # === For subclass to implement or override ===

    @abstractmethod
    def program(self, weight: Tensor, *, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight values.
                Shape: `[C_out, C_in/groups, kh, kw]`.
            bias: Per-channel integer bias added in the int64 accumulation
                domain; `None` clears any programmed bias.
                Shape: `[C_out]`.
        """
        raise NotImplementedError

    @abstractmethod
    def _conv2d_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        """Compute the convolution output, including the programmed bias."""
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _conv2d_windows(self, input: Tensor) -> Tensor:
        kh, kw = self._w_logical_shape[-2:]
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation
        x = F.pad(input, (p_w, p_w, p_h, p_h)) if p_h or p_w else input
        # Strided views preserve integer values without index tensors or indirect gathers.
        # Shape: [B, C_in, Hp, Wp] -> [B, C_in, H_out, W_out, dilated_kh, dilated_kw]
        x = x.unfold(2, (kh - 1) * d_h + 1, s_h).unfold(3, (kw - 1) * d_w + 1, s_w)
        # Shape: [B, C_in, H_out, W_out, kh, kw] -> [B, H_out, W_out, C_in, kh, kw]
        return x[..., ::d_h, ::d_w].permute(0, 2, 3, 1, 4, 5)

    @final
    def _program_int_bias(self, bias: Tensor | None, *, channels: int) -> None:
        """Store a per-output bias in the int64 accumulation domain."""
        if bias is not None and tuple(bias.shape) != (channels,):
            raise ValueError(f"require: bias.shape ({tuple(bias.shape)}) == ({channels},)")
        self._int_bias = None if bias is None else bias.long()

    def _conv2d_out_hw(self, h: int, w: int) -> tuple[int, int]:
        """Output map extent `(H_out, W_out)` for an `(h, w)` input map.

        Raises:
            ValueError: the configured geometry yields an empty output map.
        """
        kh, kw = self._w_logical_shape[-2:]
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation
        h_out = (h + 2 * p_h - d_h * (kh - 1) - 1) // s_h + 1
        w_out = (w + 2 * p_w - d_w * (kw - 1) - 1) // s_w + 1
        if h_out < 1 or w_out < 1:
            raise ValueError(f"require: positive output map; got (H_out, W_out) = {(h_out, w_out)}")
        return (h_out, w_out)
