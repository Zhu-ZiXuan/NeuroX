"""Conv2dCimUnit — CIM convolution with directly owned circuits.

See Also:
    docs/reference/architecture/unit/conv2d.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.mapping import InputActivation, InputSlotMerge
from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.primitive.digital import DigitalPolicy, SerialAccumulator
from neurox.primitive.macro.cim import CimMacro

from .base import Conv2dUnit, Conv2dUnitConfig, Conv2dUnitPolicy


class Conv2dCimUnitConfig(Conv2dUnitConfig, CimUnitConfig):
    merge: bool
    """Whether short input tiles share a macro through separate input slots."""


class Conv2dCimUnitPolicy(Conv2dUnitPolicy, CimUnitPolicy):
    pass


_Config = Conv2dCimUnitConfig
_Policy = Conv2dCimUnitPolicy


@Conv2dUnit.register_impl(config_type=_Config, policy_type=_Policy)
class Conv2dCimUnit(CimUnit, Conv2dUnit):
    """CIM-backed convolution using one programmed kernel matrix."""

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _effective_output_num: Tensor  # Shape: [merge_step, macro_group]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
        )
        output_num, input_channel, kernel_h, kernel_w = w_logical_shape
        self.tiling = self._build_tiling(input_num=input_channel * kernel_h * kernel_w, output_num=output_num)
        self.merge = InputSlotMerge(
            tile_input_num=self.tiling.tile_input_num,
            logical_tile_output_num=self.tiling.logical_tile_output_num,
            output_num=output_num,
            macro_input_num=config.cim_macro_config.input_num,
            enabled=config.merge,
        )
        self._register_nonpersistent_buffer(
            "_effective_output_num",
            self.tiling.physical_output_count(self.merge.effective_output_num(device=torch.get_default_device())),
        )
        self.cim_macro = CimMacro.from_config(
            config=config.cim_macro_config,
            policy=policy.cim_macro_policy,
            # Shape: [window=1, x_slice=1, macro_plane, in_tile, macro_group]
            inst_shape=(1, 1, self.tiling.macro_plane_num, self.tiling.in_tile_num, self.merge.macro_group_num),
            dtype=dtype,
        )
        self.w_slicer = self._build_slicer(
            slice_num=config.w_slice_num,
            encoding=config.w_slice_encoding,
            value_range=self.cim_macro.w_value_range,
        )
        self.x_slicer = self._build_slicer(
            slice_num=config.x_slice_num,
            encoding=config.x_slice_encoding,
            value_range=self.cim_macro.x_value_range,
        )
        self.input_activation = InputActivation(
            input_num=self.tiling.tile_input_num, max_active_num=self.cim_macro.max_active_num
        )
        self.phase_accumulator = SerialAccumulator(
            config=config.phase_accumulator_config,
            policy=DigitalPolicy(),
            # Shape: [macro_plane, in_tile, macro_group]
            inst_shape=(self.tiling.macro_plane_num, self.tiling.in_tile_num, self.merge.macro_group_num),
        )
        self.w_shift_adder = self._build_shift_adder(
            config=config.w_shift_adder_config,
            slicer=self.w_slicer,
            # Shape: [macro_group]
            inst_shape=(self.merge.macro_group_num,),
        )
        self.x_shift_adder = self._build_shift_adder(
            config=config.x_shift_adder_config,
            slicer=self.x_slicer,
            # Shape: [macro_group]
            inst_shape=(self.merge.macro_group_num,),
        )
        if config.padding != (0, 0):
            x_lo, x_hi = self.x_slicer.value_range
            if not (x_lo <= 0 <= x_hi):
                raise ValueError(
                    f"require: x_value_range ({(x_lo, x_hi)}) covers 0 — convolution padding injects x = 0"
                )

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.w_slicer.value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.x_slicer.value_range

    @property
    def adc_bits(self) -> int:
        return self.cim_macro.adc_bits

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        """Return the ideal-macro codes represented by one output code."""
        return self.cim_macro.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int | None) -> Tensor:
        """Combine serial window, merge, input-phase and precision reconstruction latency."""
        if len(input_shape) not in (3, 4):
            raise ValueError(
                f"latency__ns expects [input_channel, height, width] or its batched form; got {input_shape}"
            )
        h_out, w_out = self._conv2d_out_hw(*input_shape[-2:])
        window_num = h_out * w_out
        slice_num = self.x_slicer.slice_num
        merge_step_num = self.merge.merge_step_num
        phase_num = self.input_activation.input_phase_num
        port_num = self.cim_macro.output_num
        aggregated_port_num = self.tiling.logical_tile_output_num

        phase_accumulate__ns = self.phase_accumulator.latency__ns()
        w_slice_recombine__ns = self.w_shift_adder.latency__ns() if self.w_shift_adder is not None else 0.0
        x_slice_recombine__ns = self.x_shift_adder.latency__ns() if self.x_shift_adder is not None else 0.0

        # One block step retires when its phases have been accumulated.
        step_num = window_num * slice_num * merge_step_num
        access_num = step_num * phase_num
        # Shape: [merge_step, macro_group]
        macro_latency = self.cim_macro.latency__ns(
            adc_active_bits=adc_active_bits, effective_output_num=self._effective_output_num
        )
        # macro_group, in_tile and w_slice are parallel; merge_step is serial.
        # Shape: [merge_step, macro_group] -> []
        macro_latency = macro_latency.flatten(start_dim=1).amax(dim=-1).sum()
        total__ns = window_num * slice_num * phase_num * macro_latency
        total__ns += access_num * port_num * phase_accumulate__ns
        total__ns += step_num * aggregated_port_num * w_slice_recombine__ns
        # The input slices are reconstructed once every slice has arrived.
        total__ns += window_num * merge_step_num * aggregated_port_num * x_slice_recombine__ns
        return total__ns

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Flatten one copy of every output-channel kernel.

        Returns:
            Programmed kernel matrix.
            Shape: `[C_out, K]`.
        """
        # Shape: [C_out, C_in, kh, kw] -> [C_out, K]
        return weight.flatten(start_dim=1)

    @torch.no_grad()
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        # Shape: [output_channel, input_channel, kernel_h, kernel_w] -> [output, input]
        matrix = self._weight_to_matrix(weight)
        # Shape: [output, input] -> [w_slice, output, input]
        sliced = self.w_slicer.slice(matrix).movedim(-1, 0)
        # Shape: [macro_plane, in_tile, out_tile, tile_input, macro_output]
        tiled = self.tiling.map_w(sliced)
        # Shape: [macro_plane, in_tile, macro_group, macro_input, macro_output]
        packed = self.merge.map_w(tiled)
        # Shape: [window=1, x_slice=1, macro_plane, in_tile, macro_group, macro_input, macro_output]
        self.cim_macro.program(packed.unsqueeze(0).unsqueeze(0))
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        if input.ndim not in (3, 4):
            raise ValueError(f"conv2d() expects input [C_in, H, W] or [B, C_in, H, W]; got ndim {input.ndim}")
        unbatched = input.ndim == 3
        # Shape: [C_in, H, W] -> [B=1, C_in, H, W]
        x = input.unsqueeze(0) if unbatched else input
        out_hw = self._conv2d_out_hw(x.shape[-2], x.shape[-1])
        planes = self._conv2d_planes(x, out_hw=out_hw)
        y = self._matmul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        y = self._conv2d_fold(y, out_hw=out_hw)
        int_bias = self._int_bias
        if int_bias is not None:
            # Shape: [C_out] -> [C_out, H_out=1, W_out=1]
            y = y + int_bias.view(-1, 1, 1)
        if unbatched:
            # Shape: [B=1, C_out, H_out, W_out] -> [C_out, H_out, W_out]
            y = y.squeeze(0)
        return y

    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        # Shape: [..., window, input] -> [..., window, x_slice, input]
        sliced = self.x_slicer.slice(input).movedim(-1, -2)
        # Shape: [..., window, x_slice, in_tile, tile_input]
        tiled = self.tiling.map_x(sliced)
        # Shape: [..., window, x_slice, macro_plane=1, in_tile, macro_group=1, tile_input]
        organized = tiled.unsqueeze(-3).unsqueeze(-2)
        # Shape: [..., window, x_slice, macro_plane=1, in_tile, macro_group=1, input_phase, tile_input]
        phased = self.input_activation.map_x(organized)
        # Shape: [..., window, x_slice, macro_plane=1, in_tile, input_phase, tile_input]
        local = phased.squeeze(-3)
        # Shape: [..., window, x_slice, macro_plane=1, in_tile, input_phase, merge_step, macro_group, macro_input]
        routed = self.merge.map_x(local)
        # Shape: [..., merge_step, input_phase, window, x_slice, macro_plane=1, in_tile, macro_group, macro_input]
        code = routed.movedim(-3, -8).movedim(-3, -7)
        # Shape: [..., merge_step, input_phase, window, x_slice, macro_plane, in_tile, macro_group, macro_output]
        code = self.cim_macro.vec_mat_mul(
            code,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
            # Shape: [merge_step, input_phase=1, window=1, x_slice=1, macro_plane=1, in_tile=1, macro_group]
            effective_output_num=self._effective_output_num[:, None, None, None, None, None, :],
        ).long()
        # Shape: [..., merge_step, window, x_slice, macro_plane, in_tile, macro_group, macro_output]
        code = self.phase_accumulator.accumulate(code, dim=-7)
        # Shape: [..., window, x_slice, macro_plane, in_tile, merge_step, macro_group, macro_output]
        code = code.movedim(-7, -3)
        # Shape: [..., window, x_slice, macro_plane, in_tile, out_tile, macro_output]
        code = self.merge.recover(code)
        # Shape: [..., window, x_slice, w_slice, output]
        code = self.tiling.recover(code)
        # Shape: [..., window, x_slice, w_slice, output] -> [..., window, x_slice, output]
        code = self._recover_slice(code, slicer=self.w_slicer, shift_adder=self.w_shift_adder)
        # Shape: [..., window, output]
        return self._recover_slice(code, slicer=self.x_slicer, shift_adder=self.x_shift_adder)

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Gather convolution windows.

        Returns:
            Matmul-shaped input planes.
            Shape: `[B, M, K]`.
        """
        h_out, w_out = out_hw
        kh, kw = self._w_logical_shape[-2:]
        s_h, s_w = self.config.stride
        p_h, p_w = self.config.padding
        d_h, d_w = self.config.dilation
        x = input
        if p_h or p_w:
            # Shape: [B, C_in, H, W] -> [B, C_in, Hp, Wp]
            x = F.pad(x, (p_w, p_w, p_h, p_h))
        device = x.device

        # Index-grid gather rather than `F.unfold`: pure data movement, so a
        # window plane stays exact in whatever integer dtype it arrives in.
        # Shape: [H_out, kh]
        h_idx = (torch.arange(h_out, device=device) * s_h).unsqueeze(-1) + (
            torch.arange(kh, device=device) * d_h
        ).unsqueeze(0)
        # Shape: [W_out, kw]
        w_idx = (torch.arange(w_out, device=device) * s_w).unsqueeze(-1) + (
            torch.arange(kw, device=device) * d_w
        ).unsqueeze(0)
        # Shape: [B, C_in, Hp, Wp] -> [B, C_in, H_out, kh, Wp]
        x = x[..., h_idx, :]
        # Shape: [B, C_in, H_out, kh, Wp] -> [B, C_in, H_out, kh, W_out, kw]
        x = x[..., w_idx]
        # Shape: [B, C_in, H_out, kh, W_out, kw] -> [B, H_out, W_out, C_in, kh, kw]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # Shape: [B, H_out, W_out, C_in, kh, kw] -> [B, H_out, W_out, K]
        x = x.flatten(start_dim=-3)
        # Shape: [B, H_out, W_out, K] -> [B, M, K]
        return x.flatten(start_dim=-3, end_dim=-2)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Restore serial windows to the convolution output layout.

        Returns:
            Folded output map.
            Shape: `[B, C_out, H_out, W_out]`.
        """
        h_out, w_out = out_hw
        # Shape: [B, M, C_out] -> [B, H_out, W_out, C_out]
        y = output.unflatten(-2, (h_out, w_out))
        # Shape: [B, H_out, W_out, C_out] -> [B, C_out, H_out, W_out]
        folded: Tensor = y.movedim(-1, -3)
        return folded
