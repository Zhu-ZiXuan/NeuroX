"""Shared macro helpers for the example scripts."""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Literal

import torch

from neurox.common import T_ROOM__K, dataclass_from_file
from neurox.config import DEFAULT_1T1R_MACRO_TOML
from neurox.macro import NeuroxMacroQuantMatMul
from neurox.macro.xbar import XbarMacro, XbarMacroConfig
from neurox.operator import QuantSpec

XbarKind = Literal["physical", "ideal"]

# Use fp32 in the physical 1T1R path to keep solver-side arithmetic stable.
_CIRCUIT_DTYPE = torch.float32


@cache
def _macro_config(config_path: Path) -> XbarMacroConfig:
    """Cached ``[macro]`` section, polymorphically resolved via ``_neurox_type``."""
    return dataclass_from_file(XbarMacroConfig, config_path, section="macro")


def derive_quant_spec(config_path: Path = DEFAULT_1T1R_MACRO_TOML) -> QuantSpec:
    """Derive the operator quantization grid from the macro ranges."""
    macro = _build_xbar1t1r_macro(config_path, name="probe", ideal_xbar=False)
    x_qmin, x_qmax = macro.x_value_range
    w_qmin, w_qmax = macro.w_value_range
    assert w_qmin == -w_qmax, f"Expected symmetric w_value_range, got ({w_qmin}, {w_qmax})"
    return QuantSpec(x_qmin=x_qmin, x_qmax=x_qmax, w_qmax=w_qmax, y_qmin=-x_qmax, y_qmax=x_qmax)


def _build_xbar1t1r_macro(config_path: Path, *, name: str, ideal_xbar: bool) -> XbarMacro:
    """Build a 1T1R xbar macro from the given TOML."""
    macro_cfg = _macro_config(config_path)
    return XbarMacro.from_config(
        cfg=macro_cfg,
        name=name,
        T__K=T_ROOM__K,
        dtype=_CIRCUIT_DTYPE,
        ideal_xbar=ideal_xbar,
    )


def build_macro_factory(config_path: Path, *, xbar: XbarKind) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return the per-layer macro builder requested by the CLI."""
    ideal_xbar = xbar == "ideal"

    def factory(*, name: str) -> NeuroxMacroQuantMatMul:
        return _build_xbar1t1r_macro(config_path, name=name, ideal_xbar=ideal_xbar)

    return factory
