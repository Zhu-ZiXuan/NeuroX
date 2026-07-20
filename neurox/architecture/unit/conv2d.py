"""Conv2dUnit operator ABC and the ideal conv2d reference unit.

See also:
    docs/internals/architecture/unit/conv2d.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.primitive.analog.adc_common import AdcOperationPoint


class Conv2dUnit(UnitBase, ABC):
    """Operator ABC for an exact-integer drop-in replacement of ``F.conv2d``.

    Owns the conv2d template: per-call output-map geometry plus the conv
    specializations of seams 2 and 3 (:meth:`_conv2d_planes` /
    :meth:`_conv2d_fold`). The fold needs the per-call ``(H_out, W_out)``,
    so it is passed explicitly — no per-call state is ever stashed on
    ``self``. The integer bias belonging to ``F.conv2d``'s algorithmic
    scope is added in the int64 accumulation domain. Grouped convolution
    is not supported.

    Concrete hosts must be ``nn.Module`` instances and call
    :meth:`_init_conv2d_operator` in their constructor. Zero-padding
    injects ``x = 0`` activations, so a host configured with a non-zero
    padding must require ``x_value_range`` to cover ``0``.
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

    # --- conv specializations of UnitBase seams 2 and 3 (geometry-parameterized) ---

    @abstractmethod
    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Conv seam 2: ``[..., C_in, H, W]`` -> matmul-shaped planes with trailing ``[*, K_planes]``."""
        raise NotImplementedError

    @abstractmethod
    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Conv seam 3: substrate output -> ``[..., C_out, H_out, W_out]`` (undoes exactly the seam-2 axes)."""
        raise NotImplementedError

    @torch.no_grad()
    def conv2d(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Execute one integer 2-D convolution against the programmed state.

        Matches ``torch.nn.functional.conv2d`` shape semantics: the
        trailing axes are ``[C_in, H, W]`` and every leading dim is a
        broadcast batch dim passed through untouched. Template: conv
        seam 2 -> :meth:`UnitBase._matmul` -> conv seam 3, then the
        programmed integer bias (if any) is added in the int64
        accumulation domain.

        Args:
            input: Integer activation tensor with trailing ``[C_in, H, W]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer pre-requantize output tensor with trailing
            ``[C_out, H_out, W_out]``; leading dims mirror ``input``.
        """
        if input.ndim < 3:
            raise ValueError(f"conv2d() expects input with trailing [C_in, H, W]; got ndim {input.ndim}")
        out_hw = self._conv2d_out_hw(input.shape[-2], input.shape[-1])
        planes = self._conv2d_planes(input, out_hw=out_hw)
        y = self._matmul(planes, adc_operation_point=adc_operation_point)
        y = self._conv2d_fold(y, out_hw=out_hw)
        int_bias = self.int_bias
        if int_bias is not None:
            # Shape: [C_out] -> [C_out, 1, 1]; broadcast add over [..., C_out, H_out, W_out]
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

    # --- construction helper for concrete hosts ---

    def _init_conv2d_operator(
        self,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        """Store the conv geometry and register the ``int_bias`` buffer slot.

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
        self._init_int_bias_slot()


# Imported after the operator ABC definition: loading ``...cim.base``
# executes the ``cim`` package __init__, whose leaves import ``Conv2dUnit``
# from this module.
from neurox.architecture.unit.cim.base import CimUnit, CimUnitConfig, CimUnitPolicy  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class IdealConv2dUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealConv2dUnit`.

    Bring-up / reference use only — see :class:`IdealConv2dUnit`. A no-PPA
    reference: the inherited ``area_per_inst__um2`` / ``leakage_per_inst__uW``
    are supplied as ``0.0`` at construction.

    Attributes:
        x_value_range: Inclusive integer activation range (reported surface
            only; the ideal reference computes in int64 regardless).
        w_value_range: Inclusive integer weight range (reported surface
            only; the ideal reference computes in int64 regardless).
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


@dataclass(frozen=True)
class IdealConv2dUnitPolicy(CimUnitPolicy):
    """Empty policy — :class:`IdealConv2dUnit` has no nonidealities to toggle."""


@CimUnit.register_key(IdealConv2dUnitConfig)
class IdealConv2dUnit(Conv2dUnit, CimUnit):
    """Degenerate ``CimUnit``: exact-integer ``F.conv2d`` reference, substrate-free.

    Digital im2col by integer indexing plus an int64 contraction — no xbar
    tile, no slicing, no transcoding, no fp32 fast path: every contraction
    runs in int64. ``dtype``, ``T__K``, and ``ideal_xbar`` are accepted for
    API uniformity and ignored.

    Reference, not hardware: a value-domain / lossless upper-bound baseline
    with no tile, ADC, fabrication, or PPA. Use it for flow bring-up and to
    isolate QAT issues from analog modelling — never as a stand-in for a
    physical macro in a production accuracy or PPA study.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealConv2dUnitPolicy` marker.
        w_logical_shape: Logical kernel shape ``(C_out, C_in, kh, kw)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Accepted for API uniformity and ignored (no xbar tile to swap).
    """

    config: IdealConv2dUnitConfig
    nominal_weight: Tensor
    weight: Tensor

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
        self.config = config
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._init_conv2d_operator(
            kernel_size=(self._w_logical_shape[-2], self._w_logical_shape[-1]),
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )
        # 0-d nominal weight: placeholder before any ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)

    # --- value-range / ADC surface ---

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
        # ``0`` is the sentinel meaning no output quantization is applied.
        return 0

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; always ``1.0`` (no ADC)."""
        del adc_operation_point  # accepted for API uniformity
        return 1.0

    # --- lifecycle ---

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        # Shape: [C_out, C_in, kh, kw] -> [C_out, C_in*kh*kw]
        return weight.flatten(start_dim=1).to(torch.int64)

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor of shape ``(C_out, C_in, kh, kw)``.
            bias: Optional integer bias tensor of shape ``(C_out,)``;
                ``None`` clears any programmed bias.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        if weight.is_floating_point() or weight.is_complex():
            raise TypeError(f"program() expects an integer weight tensor; got dtype {weight.dtype}")
        self.weight = self._weight_to_matrix(weight.detach())
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Conv seam 2: digital im2col by pure integer indexing (dtype-agnostic data movement)."""
        h_out, w_out = out_hw
        kh, kw = self._conv2d_kernel_size
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h, d_w = self._conv2d_dilation

        x = input
        if p_h > 0 or p_w > 0:
            # Zero-fill pad on the trailing (H, W) axes only.
            # Shape: [..., C_in, H, W] -> [..., C_in, Hp, Wp]
            x = F.pad(x, (p_w, p_w, p_h, p_h))

        # Integer index grids over the padded map: output position x kernel tap.
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
    def _matmul(self, planes: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Lossless int64 contraction; ``adc_operation_point`` is unused (lossless reference)."""
        del adc_operation_point  # accepted for API uniformity
        # Shape: [..., L, C_in*kh*kw] @ [C_in*kh*kw, C_out] -> [..., L, C_out]
        return planes.to(torch.int64) @ self.weight.transpose(-2, -1)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Conv seam 3: undo the im2col axes."""
        # Shape: [..., L, C_out] -> [..., C_out, H_out, W_out]
        return output.transpose(-2, -1).unflatten(-1, out_hw)
