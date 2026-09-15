"""Build evaluation units from the bundled Xue and Ye circuit presets."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from neurox.architecture.mapping import (
    TilingMode,
)
from neurox.architecture.unit.conv2d import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy, Conv2dUnit
from neurox.architecture.unit.linear import LinearCimUnit, LinearCimUnitConfig, LinearCimUnitPolicy, LinearUnit
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import CimMacroConfig, CimMacroPolicy

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ("xue2020jssc", "ye2023jssc")


class UnitFactory:
    """Preserve preset circuit parameters while configuring logical precision.

    Digital per-operation costs are explicit evaluation assumptions, not
    measurements from either paper. Ideal mode excludes analog dynamic energy.
    """

    def __init__(
        self,
        preset: str,
        *,
        merge: bool = False,
        input_bits: int = 8,
        weight_bits: int = 8,
        accumulator_energy__fJ: float = 50.0,
        shift_energy__fJ: float = 80.0,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset: {preset}")
        self.preset = preset
        self.merge = merge
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.macro_config = CimMacroConfig.from_preset(f"works/{preset}:cim_macro")
        self.macro_policy = CimMacroPolicy.from_file(ROOT / "validations" / preset / "policy.toml", section="policy")
        self.accumulator = AccumulatorConfig(
            bit_width=32,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            latency_per_op__ns=0.0,
            energy_per_op__fJ=accumulator_energy__fJ,
        )
        self.shift = ShiftAdderConfig(
            bit_width=32,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            latency_per_op__ns=0.0,
            energy_per_op__fJ=shift_energy__fJ,
        )

    def build(
        self,
        shape: tuple[int, ...],
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        input_bits: int | None = None,
        weight_bits: int | None = None,
    ) -> LinearCimUnit | Conv2dCimUnit:
        input_bits = self.input_bits if input_bits is None else input_bits
        weight_bits = self.weight_bits if weight_bits is None else weight_bits
        macro = self.macro_config
        input_radix = macro.x_digit_r**macro.x_digit_n
        weight_radix = macro.w_digit_r**macro.w_digit_n
        common = {
            "area_per_inst__um2": 0.0,
            "leakage_per_inst__uW": 0.0,
            "cim_macro_config": macro,
            "phase_accumulator_config": self.accumulator,
            "w_slice_num": math.ceil(weight_bits / math.log2(weight_radix)),
            "w_slice_encoding": Encoding.UNSIGNED if self.preset == "ye2023jssc" else Encoding.TRUE_FORM,
            "w_shift_adder_config": self.shift,
            "tiling": TilingMode.SLICE_PLANES,
            "x_slice_num": math.ceil(input_bits / math.log2(input_radix)),
            "x_slice_encoding": Encoding.UNSIGNED,
            "x_shift_adder_config": self.shift,
        }
        unit: LinearUnit | Conv2dUnit
        if len(shape) == 2:
            unit = LinearUnit.from_config(
                config=LinearCimUnitConfig(**common),
                policy=LinearCimUnitPolicy(cim_macro_policy=self.macro_policy),
                w_logical_shape=shape,
                dtype=torch.float32,
            )
        else:
            unit = Conv2dUnit.from_config(
                config=Conv2dCimUnitConfig(
                    **common, merge=self.merge, stride=stride, padding=padding, dilation=dilation
                ),
                policy=Conv2dCimUnitPolicy(cim_macro_policy=self.macro_policy),
                w_logical_shape=shape,
                dtype=torch.float32,
            )
        if not isinstance(unit, (LinearCimUnit, Conv2dCimUnit)):
            raise TypeError(f"expected a CIM unit, got {type(unit).__name__}")
        unit.cim_macro = unit.cim_macro.to_ideal()
        return unit
