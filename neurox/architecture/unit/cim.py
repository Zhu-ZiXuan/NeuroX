"""Shared contracts and local helpers for CIM compute units."""

from __future__ import annotations

from typing import final

from torch import Tensor

from neurox.architecture.mapping import OutputSliceTiling, PlaneSliceTiling, Tiling, TilingMode
from neurox.architecture.mapping.slicer import DirectSlicer, SimpleSlicer, Slicer
from neurox.common.module import ConfigBase, PolicyBase
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, DigitalPolicy, ShiftAdder, ShiftAdderConfig
from neurox.primitive.macro.cim import CimMacroConfig, CimMacroPolicy


class CimUnitConfig(ConfigBase, base_only=True):
    # === Weight slicing ===

    w_slice_num: int
    w_slice_encoding: Encoding | None
    """Positional encoding; `None` preserves w in one unsliced digit."""

    # === Input slicing ===

    x_slice_num: int
    x_slice_encoding: Encoding | None
    """Positional encoding; `None` preserves x in one unsliced digit."""

    # === Tiling ===

    tiling: TilingMode

    # === Submodules ===

    cim_macro_config: CimMacroConfig
    phase_accumulator_config: AccumulatorConfig
    w_shift_adder_config: ShiftAdderConfig | None
    """Weight reconstruction circuit; `None` uses a functional sum without circuit cost or register wrap."""
    x_shift_adder_config: ShiftAdderConfig | None
    """Activation reconstruction circuit; `None` uses a functional sum without circuit cost or register wrap."""

    def validate(self) -> None:
        super().validate()

        # --- Weight slicing ---

        self._require_pos(self.w_slice_num, "w_slice_num")
        if self.w_slice_encoding is None and self.w_slice_num != 1:
            raise ValueError("w_slice_encoding=None requires w_slice_num=1")

        # --- Input slicing ---

        self._require_pos(self.x_slice_num, "x_slice_num")
        if self.x_slice_encoding is None and self.x_slice_num != 1:
            raise ValueError("x_slice_encoding=None requires x_slice_num=1")


class CimUnitPolicy(PolicyBase, base_only=True):
    cim_macro_policy: CimMacroPolicy


_Config = CimUnitConfig
_Policy = CimUnitPolicy


class CimUnit:
    """Local construction and precision-recovery helpers for macro-backed operators."""

    config: _Config
    policy: _Policy

    # === Tools for subclass and internal use ===

    @final
    def _build_tiling(self, *, input_num: int, output_num: int) -> Tiling:
        tiling_type = {
            TilingMode.SLICE_PLANES: PlaneSliceTiling,
            TilingMode.SLICE_OUTPUTS: OutputSliceTiling,
        }[self.config.tiling]
        return tiling_type(
            input_num=input_num,
            output_num=output_num,
            tile_input_capacity=self.config.cim_macro_config.input_num,
            tile_output_capacity=self.config.cim_macro_config.output_num,
            w_slice_num=self.config.w_slice_num,
        )

    @staticmethod
    def _build_slicer(*, slice_num: int, encoding: Encoding | None, value_range: tuple[int, int]) -> Slicer:
        if encoding is None:
            return DirectSlicer(value_range=value_range)
        return SimpleSlicer(slice_num=slice_num, slice_value_range=value_range, encoding=encoding)

    @staticmethod
    def _build_shift_adder(
        *,
        config: ShiftAdderConfig | None,
        slicer: Slicer,
        inst_shape: tuple[int, ...],
    ) -> ShiftAdder | None:
        if config is None:
            return None
        return ShiftAdder(
            config=config,
            policy=DigitalPolicy(),
            inst_shape=inst_shape,
            scale=slicer.slice_radix,
            digit_count=slicer.slice_num,
        )

    @staticmethod
    def _recover_slice(values: Tensor, *, slicer: Slicer, shift_adder: ShiftAdder | None) -> Tensor:
        """Recover one positional axis, applying the configured circuit arithmetic.

        Args:
            values: Partial results for a single operand's precision slices.
                Shape: `[..., slice, output]`.
            slicer: Numerical decomposition supplying the positional weights.
            shift_adder: Optional hardware reconstruction circuit.

        Returns:
            Reconstructed values.
            Shape: `[..., output]`.
        """
        if shift_adder is None:
            return slicer.recover(values, dim=-2)
        return shift_adder.shift_add(values, dim=-2, init_val=None)
