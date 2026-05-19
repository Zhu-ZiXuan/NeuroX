"""Shared macro helpers for the example scripts."""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch

from neurox.common import T_ROOM__K, dataclass_from_file, dict_from_file
from neurox.config import DEFAULT_1T1R_TOML
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    Requantizer,
    RequantizerConfig,
    ShiftAdder,
    ShiftAdderConfig,
)
from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.macro.xbar_macro import XbarMacro
from neurox.mapper.xbar import (
    SerialSlicer,
    SimpleMapper,
    SimpleSlicer,
    SimpleTiler,
)
from neurox.operator import QuantSpec
from neurox.xbar import Offset1T1RXbar, Offset1T1RXbarConfig

XbarKind = Literal["physical", "ideal"]

# Use fp32 in the physical 1T1R path to keep solver-side arithmetic stable.
_CIRCUIT_DTYPE = torch.float32


@cache
def _chip_config(config_path: Path) -> tuple[Offset1T1RXbarConfig, dict[str, Any]]:
    """Return ``(xbar_cfg, raw)`` for the TOML at ``config_path``."""
    xbar_cfg = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")
    raw = dict_from_file(config_path)
    return xbar_cfg, raw


def derive_quant_spec(config_path: Path = DEFAULT_1T1R_TOML) -> QuantSpec:
    """Derive the operator quantization grid from the macro ranges."""
    macro = _build_xbar1t1r_macro(config_path)
    x_qmin, x_qmax = macro.x_value_range
    w_qmin, w_qmax = macro.w_value_range
    # Weight range is expected to stay symmetric.
    assert w_qmin == -w_qmax, f"Expected symmetric w_value_range, got ({w_qmin}, {w_qmax})"
    return QuantSpec(x_qmin=x_qmin, x_qmax=x_qmax, w_qmax=w_qmax, y_qmin=-x_qmax, y_qmax=x_qmax)


def _build_xbar1t1r_macro(config_path: Path, *, name: str = "") -> XbarMacro:
    """Build a physical 1T1R ``XbarMacro`` from the given TOML."""
    xbar_cfg, raw = _chip_config(config_path)
    prefix = f"{name}." if name else ""

    xbar = Offset1T1RXbar(
        cfg=xbar_cfg,
        name=f"{prefix}xbar",
        T__K=T_ROOM__K,
        dtype=_CIRCUIT_DTYPE,
        stochastic=None,
    )

    # Macro-side digital aggregation blocks.
    accumulator_cfg = dataclass_from_file(AccumulatorConfig, config_path, section="accumulator")
    shift_adder_cfg = dataclass_from_file(ShiftAdderConfig, config_path, section="shift_adder")

    mapper = SimpleMapper(
        tiler=SimpleTiler(),
        x_slicer=SerialSlicer(
            slice_num=raw["x_transcoder"]["digit_num"],
            encoding=raw["x_transcoder"]["encoding"],
        ),
        w_slicer=SimpleSlicer(
            slice_num=raw["w_transcoder"].get("w_slice_num", 1),
            encoding=raw["w_transcoder"]["encoding"],
        ),
    )
    return XbarMacro(
        xbar=xbar,
        mapper=mapper,
        col_accumulator=Accumulator(accumulator_cfg, name=f"{prefix}col_accumulator"),
        w_shift_adder=ShiftAdder(shift_adder_cfg, name=f"{prefix}w_shift_adder"),
        x_shift_adder=ShiftAdder(shift_adder_cfg, name=f"{prefix}x_shift_adder"),
        requantizer=Requantizer(RequantizerConfig(bit_width=32), name=f"{prefix}requantizer"),
    )


def _build_xbar_ideal_macro(config_path: Path, *, name: str = "") -> XbarMacro:
    """Build the lossless twin of :func:`_build_xbar1t1r_macro`."""
    macro = _build_xbar1t1r_macro(config_path, name=name)
    macro.xbar = macro.xbar.to_ideal()
    return macro


def build_macro_factory(config_path: Path, *, xbar: XbarKind) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return the per-layer macro builder requested by the CLI."""
    builder = _build_xbar1t1r_macro if xbar == "physical" else _build_xbar_ideal_macro

    def factory(*, name: str = "") -> NeuroxMacroQuantMatMul:
        return builder(config_path, name=name)

    return factory
