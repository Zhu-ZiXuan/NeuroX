"""Shared matrix programming, execution and timing for CIM compute units.

See Also:
    docs/reference/architecture/unit/cim.md
    docs/system_design/cim_execution.md
"""

from __future__ import annotations

from abc import ABC
from typing import final

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.mapping import InputPhaseSplitter, InputSlotMerge
from neurox.architecture.mapping.slicer import DirectSlicer, SimpleSlicer, Slicer
from neurox.architecture.mapping.tiler import SimpleTiler, Tiler
from neurox.common.module import PolicyBase
from neurox.encoding import Encoding
from neurox.primitive.digital import (
    Accumulator,
    AccumulatorConfig,
    Adder,
    AdderConfig,
    DigitalPolicy,
    RadixAccumulator,
    RadixAccumulatorConfig,
    RadixSummator,
    RadixSummatorConfig,
)
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy

from .base import UnitBase, UnitConfig


class CimUnitConfig(UnitConfig, base_only=True):
    # === Weight slicing ===

    w_slice_num: int
    """Exact number of weight slices; encoding never adds slices."""
    w_slice_encoding: Encoding | None
    """Required for multiple slices. For a single direct slice, true-form and
    canonical request signed weight mapping on an unsigned macro without
    performing encoding; other selections retain the native macro range."""

    # === Input slicing ===

    x_slice_num: int
    """Exact number of input slices; encoding never adds slices."""
    x_slice_encoding: Encoding | None
    """Required for multiple input slices; ignored for a single direct slice."""

    # === Timing ===

    clock_period__ns: float
    """Digital clock period at which all supplied digital PPA values are characterized."""

    # === Submodules ===

    cim_macro_config: CimMacroConfig
    local_accumulator_config: RadixAccumulatorConfig
    """Per-local-output circuit shared by phase and input-slice recovery."""
    w_polarity_adder_config: AdderConfig | None
    """Two-input adder per simultaneously completed output pair.
    Its evaluation cost covers one complete difference, not each operand.
    `None` uses exact subtraction without circuit cost. Ignored for native mapping."""
    tile_accumulator_config: AccumulatorConfig | None
    """Per-tile-output serial accumulator; `None` uses exact tensor summation without circuit cost."""
    w_radix_summator_config: RadixSummatorConfig | None
    """Per-output weight recovery; `None` uses tensor arithmetic.
    A single slice needs no positional recovery circuit."""

    def validate(self) -> None:
        super().validate()

        # --- Weight slicing ---

        self._require_pos(self.w_slice_num, "w_slice_num")
        if self.w_slice_num > 1 and self.w_slice_encoding is None:
            raise ValueError("multiple weight slices require w_slice_encoding")

        # --- Input slicing ---

        self._require_pos(self.x_slice_num, "x_slice_num")
        if self.x_slice_num > 1 and self.x_slice_encoding is None:
            raise ValueError("multiple input slices require x_slice_encoding")

        # --- Timing ---

        self._require_pos(self.clock_period__ns, "clock_period__ns")


class CimUnitPolicy(PolicyBase, base_only=True):
    cim_macro_policy: CimMacroPolicy


_Config = CimUnitConfig
_Policy = CimUnitPolicy


class CimUnit(UnitBase, ABC, base_only=True):
    """Shared CIM implementation combined with an operator-specific unit base.

    The concrete implementation first initializes its operator base, which binds
    config and policy and initializes the common unit. It then calls this
    constructor with matrix dimensions, the input-slot sharing choice and macro
    precision. This constructor builds the CIM children and functional buffers
    without reinitializing the common unit or retaining operator metadata.

    Subclasses adapt weights to `_program_matrix`, inputs to `_vmm`, and
    results and bias to their operator layout. They compose the shared local
    and global durations into their own basic-operation latency.

    `group_num` independent matrices share mapping geometry and execute on
    separate physical circuits in parallel. A group axis always precedes matrix
    axes in programmed weights and vector inputs, including for a single matrix.
    Mapping tools preserve it; recovery never sums across groups.

    Signed weight encodings on nonnegative macros place each slice's positive
    and negative terms at adjacent output positions of `cim_macro`. Tiling uses
    the complete-pair capacity, leaving any odd final physical port unused.
    Pairing never crosses a macro or input-slot boundary. Readout differences
    precede phase and input-slice recovery; a pair spanning two scans retains
    its positive code locally until the negative code arrives.

    Construction queries native value domains through a one-instance macro
    before choosing slicing and placement. That instance is retained when it
    matches the final geometry; otherwise only the final replica population
    remains owned. Neither the query instance nor final instances are fabricated
    or programmed during construction.
    """

    config: _Config
    policy: _Policy

    tiler: Tiler
    w_slicer: Slicer
    x_slicer: Slicer

    # === Functional buffers ===

    _effective_output_num: Tensor  # Shape: [merge_step, macro_group]
    _output_enable: Tensor  # Shape: [merge_step, macro_group, tile_output]

    def __init__(
        self,
        *,
        matrix_input_num: int,
        matrix_output_num: int,
        merge: bool,
        dtype: torch.dtype,
        group_num: int = 1,
    ) -> None:
        config = self.config
        macro_config = config.cim_macro_config
        self.cim_macro = CimMacro.from_config(
            config=macro_config,
            policy=self.policy.cim_macro_policy,
            inst_shape=(1, 1, 1, 1, 1),
            dtype=dtype,
        )
        native_w_range = self.cim_macro.w_value_range
        lo, hi = native_w_range
        w_slice_range = (-hi, hi) if not macro_config.supports_signed_weights and lo == 0 else native_w_range
        if config.w_slice_num == 1:
            direct_w_range = (
                w_slice_range if config.w_slice_encoding in (Encoding.TRUE_FORM, Encoding.CANONICAL) else native_w_range
            )
            self.w_slicer = DirectSlicer(value_range=direct_w_range)
        else:
            if config.w_slice_encoding is None:
                raise ValueError("multiple weight slices require w_slice_encoding")
            self.w_slicer = SimpleSlicer(
                slice_num=config.w_slice_num,
                slice_value_range=w_slice_range,
                encoding=config.w_slice_encoding,
            )
        self._w_differential = self.w_slicer.has_signed_slices and not macro_config.supports_signed_weights
        if self._w_differential and (native_w_range[0] != 0 or native_w_range[1] < 1):
            raise ValueError(f"differential weights require a physical range [0, U] with U > 0; got {native_w_range}")
        polarity_num = 2 if self._w_differential else 1
        output_per_tile = macro_config.output_num // polarity_num
        if output_per_tile == 0:
            raise ValueError("differential weights require at least two macro output positions")
        self.local_lane_num = min(output_per_tile, -(-macro_config.lane_num // polarity_num))
        output_tile_num = -(-config.w_slice_num * matrix_output_num // output_per_tile)
        self.tile_accumulator = None
        if config.tile_accumulator_config is not None:
            self.tile_accumulator = Accumulator(
                config=config.tile_accumulator_config,
                policy=DigitalPolicy(),
                inst_shape=(group_num, output_tile_num, output_per_tile),
            )
        self.tiler = SimpleTiler(
            matrix_input_num=matrix_input_num,
            matrix_output_num=matrix_output_num,
            input_per_tile=min(matrix_input_num, config.cim_macro_config.input_num),
            output_per_tile=output_per_tile,
            w_slice_num=config.w_slice_num,
            recovery_circuit=self.tile_accumulator,
        )
        self.merge = InputSlotMerge(
            input_per_tile=self.tiler.input_per_tile,
            output_tile_num=self.tiler.output_tile_num,
            macro_input_num=config.cim_macro_config.input_num,
            enabled=merge,
        )
        # Shape: [group, in_tile, input_phase=1, merge_step=1, macro_group]
        macro_inst_shape = (group_num, self.tiler.input_tile_num, 1, 1, self.merge.macro_group_num)
        if self.cim_macro.inst_shape != macro_inst_shape:
            self.cim_macro = CimMacro.from_config(
                config=macro_config,
                policy=self.policy.cim_macro_policy,
                inst_shape=macro_inst_shape,
                dtype=dtype,
            )
        self.w_polarity_adder = None
        if self._w_differential and config.w_polarity_adder_config is not None:
            self.w_polarity_adder = Adder(
                config=config.w_polarity_adder_config,
                policy=DigitalPolicy(),
                inst_shape=(*macro_inst_shape, self.local_lane_num),
            )
        self.w_radix_summator = None
        if self.w_slicer.slice_num > 1 and config.w_radix_summator_config is not None:
            self.w_radix_summator = RadixSummator(
                config=config.w_radix_summator_config,
                policy=DigitalPolicy(),
                inst_shape=(group_num, matrix_output_num),
            )
        self.w_slicer.recovery_circuit = self.w_radix_summator
        self.local_accumulator = RadixAccumulator(
            config=config.local_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(*macro_inst_shape, self.local_lane_num),
        )
        if config.x_slice_num == 1:
            self.x_slicer = DirectSlicer(value_range=self.cim_macro.x_value_range)
        else:
            if config.x_slice_encoding is None:
                raise ValueError("multiple input slices require x_slice_encoding")
            self.x_slicer = SimpleSlicer(
                slice_num=config.x_slice_num,
                slice_value_range=self.cim_macro.x_value_range,
                encoding=config.x_slice_encoding,
                recovery_circuit=self.local_accumulator,
            )
        self.input_phase_splitter = InputPhaseSplitter(
            input_num=self.tiler.input_per_tile,
            max_active_num=self.cim_macro.max_active_num,
            accumulator=self.local_accumulator,
        )

        # Shape: [out_tile]
        effective_output_num = self.tiler.effective_output_num(device=torch.get_default_device())
        # Shape: [merge_step, macro_group]
        logical_output_num = self.merge.map_effective_output_num(effective_output_num)
        self._register_nonpersistent_buffer("_effective_output_num", logical_output_num * polarity_num)
        self._register_nonpersistent_buffer(
            "_output_enable",
            torch.arange(output_per_tile) < logical_output_num.unsqueeze(-1),
        )

    # === Required by base class ===

    @property
    @final
    def w_value_range(self) -> tuple[int, int]:
        return self.w_slicer.value_range

    @property
    @final
    def x_value_range(self) -> tuple[int, int]:
        return self.x_slicer.value_range

    @property
    @final
    def adc_bits(self) -> int:
        return self.cim_macro.adc_bits

    @final
    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        return self.cim_macro.rescale_factor(quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)

    # === Tools for subclass and internal use ===

    @final
    @torch.no_grad()
    def _program_matrix(self, weight: Tensor) -> None:
        """Program the matrix geometry supplied at construction.

        Slice signed weights before decomposing each slice into nonnegative
        terms. Placement keeps each positive/negative pair adjacent.

        Args:
            weight: Integer weights in the unit's logical value range.
                Shape: `[group, output, input]`.
        """
        # Shape: [..., output, input] -> [..., w_slice, output, input]
        sliced = self.w_slicer.slice(weight, dim=-3)
        if self._w_differential:
            # Shape: [...] -> [polarity, ...]
            sliced = torch.stack((sliced.clamp_min(0), (-sliced).clamp_min(0)), dim=0)
        # Leading polarity, when present, passes through tiling and placement.
        # Shape: [..., in_tile, out_tile, tile_in, tile_output]
        tiled = self.tiler.map_w(sliced)
        # Shape: [..., in_tile, macro_group, macro_input, tile_output]
        packed = self.merge.map_w(tiled)
        if self._w_differential:
            # Shape: [polarity, ..., tile_output] -> [..., tile_output*2]
            packed = packed.movedim(0, -1).flatten(-2)
        # An odd physical output capacity leaves one unpaired port disabled.
        packed = F.pad(packed, (0, self.cim_macro.output_num - packed.shape[-1]))
        # Shape: [..., in_tile, input_phase=1, merge_step=1, macro_group, macro_input, macro_output]
        packed = packed.unsqueeze(-4).unsqueeze(-5)
        self.cim_macro.program(packed)

    @final
    def _vmm(self, input: Tensor, *, quantization_mode: int, adc_active_bits: int | None) -> Tensor:
        """Evaluate vectors against the programmed matrix without bias.

        Every leading axis passes through unchanged. The group axis selects
        the independently programmed matrix.

        Args:
            input: Integer vectors in the unit's logical value range.
                Shape: `[..., group, input]`.

        Returns:
            Reconstructed outputs preserving all leading axes.
            Shape: `[..., group, output]`.
        """
        # Shape: [..., group, input] -> [..., x_slice, group, input]
        sliced = self.x_slicer.slice(input, dim=-3)
        # Shape: [..., x_slice, group, in_tile, tile_in]
        tiled = self.tiler.map_x(sliced)
        # Shape: [..., x_slice, group, in_tile, input_phase, tile_in]
        phased = self.input_phase_splitter.split(tiled, dim=-2)
        # Shape: [..., x_slice, group, in_tile, input_phase, merge_step, macro_group, macro_input]
        routed = self.merge.map_x(phased)
        # Shape: [..., x_slice, group, in_tile, input_phase, merge_step, macro_group, macro_output]
        code = self.cim_macro.vec_mat_mul(
            routed,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
            effective_output_num=self._effective_output_num,
        ).long()
        if self._w_differential:
            paired_num = 2 * self.tiler.output_per_tile
            # Shape: [..., macro_output] -> [..., tile_output]
            positive = code[..., :paired_num:2]
            negative = code[..., 1:paired_num:2]
            if self.w_polarity_adder is None:
                code = (positive - negative).where(self._output_enable, 0)
            else:
                code = self.w_polarity_adder.add(positive, -negative, enable=self._output_enable)
        # Shape: [..., x_slice, group, in_tile, merge_step, macro_group, tile_output]
        code = self.input_phase_splitter.recover(code, dim=-4, enable=self._output_enable)
        # Shape: [..., group, in_tile, merge_step, macro_group, tile_output]
        code = self.x_slicer.recover(code, dim=-6, enable=self._output_enable)
        # Shape: [..., group, in_tile, out_tile, tile_output]
        code = self.merge.recover(code)
        # Shape: [..., group, w_slice, output]
        code = self.tiler.recover(code)
        # Shape: [..., group, output]
        return self.w_slicer.recover(code, dim=-2)

    @final
    def _vmm_local_latency__ns(self, *, adc_active_bits: int | None) -> float:
        """Complete one vector's macro accesses and local digital updates.

        Local updates are assumed to keep pace with analog scans. Polarity
        recovery and local accumulation share one cycle. Each access waits for
        its final local update without clock-edge alignment or backpressure.
        """
        clock_period__ns = self.config.clock_period__ns
        input_phase_num = self.input_phase_splitter.input_phase_num
        input_slice_num = self.x_slicer.slice_num

        # Shape: [merge_step, macro_group]
        macro__ns = self.cim_macro.latency__ns(
            adc_active_bits=adc_active_bits,
            effective_output_num=self._effective_output_num,
        )
        # Shape: [merge_step, macro_group]
        local__ns = macro__ns + (self._effective_output_num > 0) * clock_period__ns
        # Shape: [merge_step, macro_group] -> []
        local__ns = local__ns.amax(dim=-1).sum()
        return float(local__ns) * input_slice_num * input_phase_num

    @final
    def _vmm_global_latency__ns(self) -> float:
        """Transfer and process one vector's completed local results.

        One complete input-tile row arrives per cycle and uses one subsequent
        accumulation cycle. Configured weight-slice reconstruction follows the
        final tile update as a separate stage.
        """
        w_cycles = 0 if self.w_radix_summator is None else 1
        return (self.tiler.input_tile_num + 1 + w_cycles) * self.config.clock_period__ns
