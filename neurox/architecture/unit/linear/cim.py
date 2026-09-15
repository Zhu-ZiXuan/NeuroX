"""CIM linear operator with directly owned macro and reconstruction circuits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.mapping import InputActivation
from neurox.architecture.unit.cim import CimUnit, CimUnitConfig, CimUnitPolicy
from neurox.primitive.digital import DigitalPolicy, SerialAccumulator
from neurox.primitive.macro.cim import CimMacro

from .base import LinearUnit, LinearUnitConfig, LinearUnitPolicy


class LinearCimUnitConfig(LinearUnitConfig, CimUnitConfig):
    pass


class LinearCimUnitPolicy(LinearUnitPolicy, CimUnitPolicy):
    pass


_Config = LinearCimUnitConfig
_Policy = LinearCimUnitPolicy


@LinearUnit.register_impl(config_type=_Config, policy_type=_Policy)
class LinearCimUnit(CimUnit, LinearUnit):
    """CIM linear operator using separate physical tiles without input-slot sharing."""

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _effective_output_num: Tensor  # Shape: [out_tile]

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
        output_num, input_num = w_logical_shape
        self.tiling = self._build_tiling(input_num=input_num, output_num=output_num)
        # Shape: [out_tile]
        output_starts = torch.arange(self.tiling.out_tile_num) * self.tiling.logical_tile_output_num
        counts = (output_num - output_starts).clamp(0, self.tiling.logical_tile_output_num)
        self._register_nonpersistent_buffer("_effective_output_num", self.tiling.physical_output_count(counts))
        self.cim_macro = CimMacro.from_config(
            config=config.cim_macro_config,
            policy=policy.cim_macro_policy,
            # Shape: [vector=1, x_slice=1, macro_plane, in_tile, out_tile]
            inst_shape=(1, 1, self.tiling.macro_plane_num, self.tiling.in_tile_num, self.tiling.out_tile_num),
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
            # Shape: [macro_plane, in_tile, out_tile]
            inst_shape=(self.tiling.macro_plane_num, self.tiling.in_tile_num, self.tiling.out_tile_num),
        )
        self.w_shift_adder = self._build_shift_adder(
            config=config.w_shift_adder_config,
            slicer=self.w_slicer,
            # Shape: [out_tile]
            inst_shape=(self.tiling.out_tile_num,),
        )
        self.x_shift_adder = self._build_shift_adder(
            config=config.x_shift_adder_config,
            slicer=self.x_slicer,
            # Shape: [out_tile]
            inst_shape=(self.tiling.out_tile_num,),
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

    @torch.no_grad()
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        # Shape: [output, input] -> [w_slice, output, input]
        sliced = self.w_slicer.slice(weight).movedim(-1, 0)
        # Shape: [macro_plane, in_tile, out_tile, tile_input, macro_output]
        tiled = self.tiling.map_w(sliced)
        # Shape: [vector=1, x_slice=1, macro_plane, in_tile, out_tile, macro_input, macro_output]
        programmed = (
            F.pad(tiled, (0, 0, 0, self.cim_macro.input_num - self.tiling.tile_input_num)).unsqueeze(0).unsqueeze(0)
        )
        self.cim_macro.program(programmed)
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _linear_impl(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        # Shape: [..., input] -> [..., vector=1, input]
        planes = input.unsqueeze(-2)
        # Shape: [..., vector=1, output] -> [..., output]
        output = self._matmul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits).squeeze(-2)
        if self._int_bias is not None:
            output = output + self._int_bias
        return output

    def _matmul(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        # Shape: [..., vector, input] -> [..., vector, x_slice, input]
        sliced = self.x_slicer.slice(input).movedim(-1, -2)
        # Shape: [..., vector, x_slice, in_tile, tile_input]
        tiled = self.tiling.map_x(sliced)
        # Shape: [..., vector, x_slice, in_tile, input_phase, tile_input]
        phased = self.input_activation.map_x(tiled)
        # Shape: [..., input_phase, vector, x_slice, macro_plane=1, in_tile, out_tile=1, tile_input]
        phased = phased.movedim(-2, -5).unsqueeze(-3).unsqueeze(-2)
        # Shape: [..., input_phase, vector, x_slice, macro_plane=1, in_tile, out_tile=1, macro_input]
        driven = F.pad(phased, (0, self.cim_macro.input_num - self.tiling.tile_input_num))
        # Shape: [..., input_phase, vector, x_slice, macro_plane, in_tile, out_tile, macro_output]
        code = self.cim_macro.vec_mat_mul(
            driven,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
            # Shape: [vector=1, x_slice=1, macro_plane=1, in_tile=1, out_tile]
            effective_output_num=self._effective_output_num.view(1, 1, 1, 1, -1),
        ).long()
        # Shape: [..., vector, x_slice, macro_plane, in_tile, out_tile, macro_output]
        code = self.phase_accumulator.accumulate(code, dim=-7)
        # Shape: [..., vector, x_slice, w_slice, output]
        code = self.tiling.recover(code)
        # Shape: [..., vector, x_slice, output]
        code = self._recover_slice(code, slicer=self.w_slicer, shift_adder=self.w_shift_adder)
        # Shape: [..., vector, output]
        return self._recover_slice(code, slicer=self.x_slicer, shift_adder=self.x_shift_adder)

    def latency__ns(self, input_shape: tuple[int, ...], *, adc_active_bits: int | None) -> Tensor:
        """Return one linear evaluation's latency; leading batch dimensions are parallel."""
        slice_num = self.x_slicer.slice_num
        phase_num = self.input_activation.input_phase_num
        macro_latency = self.cim_macro.latency__ns(
            adc_active_bits=adc_active_bits,
            effective_output_num=self._effective_output_num,
        )
        # Shape: [out_tile] -> []
        macro_latency = macro_latency.amax()
        phase_latency = self.cim_macro.output_num * self.phase_accumulator.latency__ns()
        w_latency = self.w_shift_adder.latency__ns() if self.w_shift_adder is not None else 0.0
        x_latency = self.x_shift_adder.latency__ns() if self.x_shift_adder is not None else 0.0
        return slice_num * phase_num * (macro_latency + phase_latency) + self.tiling.logical_tile_output_num * (
            slice_num * w_latency + x_latency
        )
