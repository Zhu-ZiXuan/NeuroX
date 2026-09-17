"""Conv2dUnit operator interface.

See Also:
    docs/reference/architecture/unit/conv2d.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase, UnitConfig
from neurox.common.module import PolicyBase
from neurox.common.registry_mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealConv2dUnit


class Conv2dUnitConfig(UnitConfig, ABC):
    # === Convolution geometry ===

    stride: tuple[int, int]
    """Output step `(s_h, s_w)`."""
    padding: tuple[int, int]
    """Zero-pad extent `(p_h, p_w)` on each side."""
    dilation: tuple[int, int]
    """Kernel tap spacing `(d_h, d_w)`."""

    def validate(self) -> None:
        super().validate()

        # --- Convolution geometry ---

        self._require_pos(self.stride[0], "stride[0]")
        self._require_pos(self.stride[1], "stride[1]")
        self._require_non_neg(self.padding[0], "padding[0]")
        self._require_non_neg(self.padding[1], "padding[1]")
        self._require_pos(self.dilation[0], "dilation[0]")
        self._require_pos(self.dilation[1], "dilation[1]")


class Conv2dUnitPolicy(PolicyBase, ABC):
    pass


_Config = Conv2dUnitConfig
_Policy = Conv2dUnitPolicy


class Conv2dUnit(RegistryMixin[_Config, _Policy], UnitBase, ABC):
    """Interface for an integer `torch.nn.functional.conv2d` replacement.

    Grouped convolution is not supported.
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
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        self._w_logical_shape = tuple(w_logical_shape)
        self._dtype = dtype
        if len(w_logical_shape) != 4:
            raise ValueError("conv2d weight shape must have 4 axes")

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Conv2dUnit:
        """Construct the conv2d implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(config=config, policy=policy, w_logical_shape=w_logical_shape, dtype=dtype)

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
            area_per_inst__um2=self.area__um2,
            leakage_per_inst__uW=self.leakage__uW,
            stride=self.config.stride,
            padding=self.config.padding,
            dilation=self.config.dilation,
        )
        ideal = IdealConv2dUnit(
            config=config,
            policy=IdealConv2dUnitPolicy(),
            w_logical_shape=self._w_logical_shape,
            dtype=self._dtype,
        )
        ideal.set_temperature(self.T__K)
        return ideal

    @final
    @torch.no_grad()
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
                `[C_in, H, W]` form, or the configured geometry yields an empty
                output map.
        """
        return self._conv2d_impl(input, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    # === For subclass to implement or override ===

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight values.
                Shape: `[C_out, C_in, kh, kw]`.
            bias: Per-channel integer bias added in the int64 accumulation
                domain; `None` clears any programmed bias.
                Shape: `[C_out]`.
        """
        raise NotImplementedError

    @abstractmethod
    def _conv2d_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        """Implement the convolution operation, including the programmed bias."""
        raise NotImplementedError

    # === Tools for subclass and internal use ===

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
        s_h, s_w = self.config.stride
        p_h, p_w = self.config.padding
        d_h, d_w = self.config.dilation
        h_out = (h + 2 * p_h - d_h * (kh - 1) - 1) // s_h + 1
        w_out = (w + 2 * p_w - d_w * (kw - 1) - 1) // s_w + 1
        if h_out < 1 or w_out < 1:
            raise ValueError(f"require: positive output map; got (H_out, W_out) = {(h_out, w_out)}")
        return (h_out, w_out)
