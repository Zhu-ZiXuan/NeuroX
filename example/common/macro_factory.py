"""Shared macro helpers for the example scripts.

The macro config TOML is polymorphically resolved via ``_neurox_type``;
the same factory works for the registered XbarMacro family members:

- ``DirectXbarMacroConfig`` / ``InterArraySliceXbarMacroConfig`` /
  ``IntraArraySliceXbarMacroConfig`` (xbar-using) with either a physical
  or an :class:`IdealXbarConfig` inside ``xbar_cfg``.
- ``IdealXbarMacroConfig`` (degenerate; no xbar — pure integer matmul baseline).
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

from neurox.common import T_ROOM__K, dataclass_from_file
from neurox.macro import NeuroxMacroQuantMatMul
from neurox.macro.xbar import IdealXbarMacroConfig, XbarMacro, XbarMacroConfig
from neurox.operator import QuantSpec
from neurox.xbar import IdealXbarConfig

# Use fp32 in the physical 1T1R path to keep solver-side arithmetic stable.
_CIRCUIT_DTYPE = torch.float32

# Operator construction passes its own weight shape into the factory; the
# probe path needs a non-empty placeholder just to read value ranges.
_PROBE_W_LOGICAL_SHAPE = (1, 1)


@cache
def read_macro_config(config_path: Path) -> XbarMacroConfig:
    """Cached ``[macro]`` section, polymorphically resolved via ``_neurox_type``."""
    return dataclass_from_file(XbarMacroConfig, config_path, section="macro")


def supports_xbar_override(cfg: XbarMacroConfig) -> bool:
    """Whether ``--xbar physical|ideal`` is meaningful for ``cfg``.

    True only when the config carries a physical xbar (i.e. is xbar-using
    and ``cfg.xbar_cfg`` is not already an :class:`IdealXbarConfig`).
    """
    if isinstance(cfg, IdealXbarMacroConfig):
        return False
    xbar_cfg = getattr(cfg, "xbar_cfg", None)
    if xbar_cfg is None:
        return False
    return not isinstance(xbar_cfg, IdealXbarConfig)


def derive_quant_spec(config_path: Path) -> QuantSpec:
    """Derive the operator quantization grid from the macro ranges."""
    macro = _build_macro(
        config_path,
        name="probe",
        w_logical_shape=_PROBE_W_LOGICAL_SHAPE,
        ideal_xbar=False,
    )
    x_qmin, x_qmax = macro.x_value_range
    w_qmin, w_qmax = macro.w_value_range
    assert w_qmin == -w_qmax, f"Expected symmetric w_value_range, got ({w_qmin}, {w_qmax})"
    return QuantSpec(x_qmin=x_qmin, x_qmax=x_qmax, w_qmax=w_qmax, y_qmin=-x_qmax, y_qmax=x_qmax)


def _build_macro(
    config_path: Path,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
    ideal_xbar: bool,
) -> NeuroxMacroQuantMatMul:
    """Build one macro from a TOML config, dispatching on the cfg type.

    ``ideal_xbar`` is honoured only by xbar-using members (swaps the
    physical xbar for its ideal twin); :class:`IdealXbarMacro` ignores it.
    """
    cfg = read_macro_config(config_path)
    return XbarMacro.from_config(
        cfg=cfg,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=_CIRCUIT_DTYPE,
        T__K=T_ROOM__K,
        ideal_xbar=ideal_xbar,
    )


def build_macro_factory(config_path: Path, *, ideal_xbar: bool) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return the per-layer macro builder for the requested TOML + xbar flavour.

    Args:
        config_path: TOML path resolved against ``example/<model>/`` by the caller.
        ideal_xbar: Forwarded to the macro; relevant only when the config
            carries a physical xbar (see :func:`supports_xbar_override`).
    """

    def factory(*, name: str, w_logical_shape: tuple[int, ...]) -> NeuroxMacroQuantMatMul:
        return _build_macro(
            config_path,
            name=name,
            w_logical_shape=w_logical_shape,
            ideal_xbar=ideal_xbar,
        )

    return factory


__all__ = [
    "build_macro_factory",
    "derive_quant_spec",
    "read_macro_config",
    "supports_xbar_override",
]
