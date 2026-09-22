"""Load model-local macro overrides and unit configurations for evaluation."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import torch

from neurox.architecture.unit.conv2d import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
from neurox.architecture.unit.linear import LinearCimUnit, LinearCimUnitConfig, LinearCimUnitPolicy

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ("xue2020jssc", "ye2023jssc")


class UnitFactory:
    """Construct independent units using the selected model directory's files.

    Immutable configurations may be shared; each build creates fresh hardware.
    Convolution geometry is passed at construction from the source layer.
    An explicit merge override replaces only that mapping choice. Circuit
    parameters, precision slicing, and policies come from files.
    """

    def __init__(
        self,
        config_dir: Path,
        preset: str,
        *,
        ideal_macro: bool = False,
        merge: bool | None = None,
    ) -> None:
        self.config_dir = config_dir.resolve()
        with (self.config_dir / "model.toml").open("rb") as stream:
            settings = tomllib.load(stream)["evaluation"]
        if not settings["enabled"]:
            raise ValueError(settings["reason"])
        if preset not in PRESETS:
            raise ValueError(f"unknown preset: {preset}")
        self.preset = preset
        self.ideal_macro = ideal_macro
        self.input_bits: int = settings["input_bits"]
        self.weight_bits: int = settings["weight_bits"]
        self.quantized_linear_input: str = settings["quantized_linear_input"]
        self.config_file = self.config_dir / f"{preset}.toml"
        self._linear_configs = {
            profile: LinearCimUnitConfig.from_file(self.config_file, section=f"{profile}.linear")
            for profile in settings["profiles"]
        }
        self._conv2d_configs = {
            profile: Conv2dCimUnitConfig.from_file(self.config_file, section=f"{profile}.conv2d")
            for profile in settings["profiles"]
        }
        self._linear_policy = LinearCimUnitPolicy.from_file(self.config_file, section="linear_policy")
        self._conv2d_policy = Conv2dCimUnitPolicy.from_file(self.config_file, section="conv2d_policy")
        self._merge_override = merge
        self.merge = next(iter(self._conv2d_configs.values())).merge if merge is None else merge
        common = next(iter(self._linear_configs.values()))
        self.local_accumulator = common.local_accumulator_config
        self.tile_accumulator = common.tile_accumulator_config
        self.radix_summator = common.w_radix_summator_config
        self.polarity_adder = common.w_polarity_adder_config

    def supports_precision(self, *, input_bits: int, weight_bits: int) -> bool:
        """Check whether the local files declare this logical precision."""
        return f"w{weight_bits}a{input_bits}" in self._linear_configs

    def build_linear(
        self,
        shape: tuple[int, ...],
        *,
        input_bits: int | None = None,
        weight_bits: int | None = None,
    ) -> LinearCimUnit:
        """Construct one unit for the complete logical weight matrix."""
        input_bits = self.input_bits if input_bits is None else input_bits
        weight_bits = self.weight_bits if weight_bits is None else weight_bits
        config = self._linear_configs[f"w{weight_bits}a{input_bits}"]
        unit = LinearCimUnit(config=config, policy=self._linear_policy, w_logical_shape=shape, dtype=torch.float32)
        if self.ideal_macro:
            unit.cim_macro = unit.cim_macro.to_ideal()
        return unit

    def build_conv2d(
        self,
        shape: tuple[int, ...],
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        input_bits: int | None = None,
        weight_bits: int | None = None,
    ) -> Conv2dCimUnit:
        """Load circuit configuration and bind the original convolution geometry."""
        input_bits = self.input_bits if input_bits is None else input_bits
        weight_bits = self.weight_bits if weight_bits is None else weight_bits
        base = self._conv2d_configs[f"w{weight_bits}a{input_bits}"]
        config = base if self._merge_override is None else replace(base, merge=self._merge_override)
        unit = Conv2dCimUnit(
            config=config,
            policy=self._conv2d_policy,
            w_logical_shape=shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=torch.float32,
        )
        if self.ideal_macro:
            unit.cim_macro = unit.cim_macro.to_ideal()
        return unit
