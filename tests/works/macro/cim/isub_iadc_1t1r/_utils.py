"""Shared build / probe / reference helpers for the isub_iadc_1t1r scheme tests.

Every solve-bearing test builds through :func:`build_tile` (config -> registry
dispatch -> ``.to(device)`` -> ``.eval()`` -> ``.fabricate()``), expands
activations into zero-masked WL planes via :func:`masked_planes` (the engine
mask formula — the macro consumes pre-expanded planes and adds no axis of its
own), and checks the per-plane ADC codes against
:func:`per_phase_clamp_reference` — the CPU int64 unit-role reference
``clamp(x_p . w_p, -7, 7)`` per WL sub-phase.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
from pathlib import Path

import torch
from torch import Tensor

from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.array import XbarArraySteadyState
from neurox.works.macro.cim.isub_iadc_1t1r.macro import (
    IsubIadc1t1rCimMacro,
    IsubIadc1t1rCimMacroConfig,
    IsubIadc1t1rCimMacroPolicy,
)

PKG_DIR = Path(str(importlib.resources.files("neurox.works.macro.cim.isub_iadc_1t1r")))
CONFIG_PATH = PKG_DIR / "params" / "default.toml"
POLICY_PATH = PKG_DIR / "policy" / "all_off.toml"
TINY_OVERLAY_PATH = Path(__file__).resolve().parent / "tiny_xbar.toml"

MAG_MAX = (1 << 3) - 1  # 7 — the tiny fixture's 3-bit magnitude saturation
ADC_OP = AdcOperationPoint(adc_mode=0, adc_bits=3)  # tiny fixture operating point

# Tiny overlay geometry (tests/tiny_xbar.toml): P = 2 phases, n_lane = 2.
TINY_COL_NUM = 2
TINY_ROW_NUM = 16
TINY_ACTIVE_ROW_NUM = 8
TINY_PHASE_NUM = 2

# Canonical default geometry (params/default.toml): 5-bit magnitude + sign,
# two operating modes (signed / unsigned deployment layer groups).
DEF_COL_NUM = 256
DEF_ROW_NUM = 256
DEF_ACTIVE_ROW_NUM = 64
DEF_PHASE_NUM = 4
DEF_MAG_MAX = (1 << 5) - 1  # 31 — the canonical 5-bit magnitude saturation
DEF_ADC_BITS = 5
DEF_ADC_MODE_NUM = 2


def tile_device(xbar: IsubIadc1t1rCimMacro) -> torch.device:
    """Device the tile lives on (first buffer of the module tree)."""
    return next(xbar.buffers()).device


def masked_planes(x: Tensor, *, row_num: int, max_active_rows: int, inst_rank: int = 0) -> Tensor:
    """Zero-masked WL planes via the engine mask formula.

    Shape: [..., *span, row_num] -> [..., P, *span, row_num]; the P axis
    inserts immediately left of the ``inst_rank``-wide inst-aligned span and
    plane ``p`` keeps exactly rows ``[p*max_active_rows, (p+1)*max_active_rows)``,
    zeros elsewhere (WL off).
    """
    p_num = row_num // max_active_rows
    mask = torch.arange(row_num, device=x.device) // max_active_rows == torch.arange(p_num, device=x.device).unsqueeze(
        -1
    )
    # Shape: [P, row_num] -> [P, 1*inst_rank, row_num]
    mask = mask.reshape(p_num, *(1,) * inst_rank, row_num)
    # Shape: [..., *span, row_num] -> [..., P, *span, row_num]
    return torch.where(mask, x.unsqueeze(max(-(inst_rank + 2), -(x.ndim + 1))), x.new_zeros(()))


def load_config(*config_paths: Path) -> IsubIadc1t1rCimMacroConfig:
    """Load the scheme config from ``config_paths`` (first-wins deep merge)."""
    config = CimMacroConfig.from_file(*config_paths, section="cim_macro")
    assert isinstance(config, IsubIadc1t1rCimMacroConfig)
    return config


def load_all_off_policy() -> IsubIadc1t1rCimMacroPolicy:
    """Load the canonical all-off (lossless baseline) policy."""
    policy = CimMacroPolicy.from_file(POLICY_PATH, section="policy")
    assert isinstance(policy, IsubIadc1t1rCimMacroPolicy)
    return policy


def build_tile(
    config: IsubIadc1t1rCimMacroConfig,
    *,
    device: torch.device | None = None,
    policy: IsubIadc1t1rCimMacroPolicy | None = None,
    seed: int | None = None,
    inst_shape: tuple[int, ...] = (),
) -> IsubIadc1t1rCimMacro:
    """Build + fabricate one tile on ``device`` (all-off policy unless given)."""
    if policy is None:
        policy = load_all_off_policy()
    xbar = CimMacro.from_config(
        config=config,
        policy=policy,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(xbar, IsubIadc1t1rCimMacro)
    if device is not None:
        xbar.to(device)
    xbar.eval()
    if seed is not None:
        torch.manual_seed(seed)
    xbar.fabricate()
    return xbar


def with_ref_levels(
    config: IsubIadc1t1rCimMacroConfig, ref_levels__uA: tuple[float, ...]
) -> IsubIadc1t1rCimMacroConfig:
    """Install one flat ladder on BOTH threshold copies (kept consistent).

    The flat tuple canonicalizes to a single-mode 2-D row on both configs, so
    the per-mode consistency law keeps holding.
    """
    return dataclasses.replace(
        config,
        adc_config=dataclasses.replace(config.adc_config, ref_levels__uA=tuple(ref_levels__uA)),
        reference_config=dataclasses.replace(config.reference_config, i_refs__uA=tuple(ref_levels__uA)),
    )


def array_read(xbar: IsubIadc1t1rCimMacro, x_planes: Tensor) -> XbarArraySteadyState:
    """Boundary drive + array steady state for WL planes ``[..., row_num]``.

    Reproduces the ``vec_mat_mul`` array stage: WL DAC convert at the
    weight-grid full leading, one clamp-reference snapshot, and the kernel
    ``solve_array`` with the macro's lane-grouped BL clamp adapter.
    """
    _phys_col_num, row_num = xbar.core.weight_grid_shape[-2:]
    leading = torch.broadcast_shapes(xbar.core.weight_grid_shape, x_planes.unsqueeze(-2).shape)[:-2]
    with torch.no_grad():
        v_wl = xbar.wl_dac.convert(x_planes.expand(*leading, row_num))
        clamp_taps = xbar.clamp_ref.v_ref__V(xbar.clamp_ref.snapshot())
        return xbar.core.solve_array(
            v_wl,
            bl_driver=xbar._bl_clamp_lanes,
            bl_v_ref__V=clamp_taps[0],
            sl_driver=xbar.sl_driver,
            sl_v_ref__V=clamp_taps[1],
            t_conduct__ns=xbar.config.t_conduct__ns,
        )


def probe_i_sub(xbar: IsubIadc1t1rCimMacro, x_planes: Tensor) -> tuple[Tensor, Tensor]:
    """Analog ``(i_sub, sign)`` at the ADC input for WL planes ``[..., row_num]``.

    Reproduces the ``vec_mat_mul`` analog chain up to the ADC input (boundary
    drive + array solve -> polarity split + serial axes to leading ->
    p-mirror -> n-mirror -> subtractor), then moves the serial axes back
    trailing so the probe returns ``[..., n_io, io_col_num]`` with the
    columns of each IO in logical order. Leading dims of ``x_planes`` (batch
    and/or the sub-phase axis) ride through unchanged.
    """
    cfg = xbar.config
    lanes_per_io = cfg.io_col_num // cfg.mux_factor
    with torch.no_grad():
        steady = array_read(xbar, x_planes)
        i_lane = (
            steady.i_bl_port__uA.unflatten(-1, (xbar.col_num, 2))
            .movedim(-1, -2)
            .unflatten(-1, (xbar.n_lane, cfg.mux_factor))
            .movedim(-1, 0)
        )
        i_wdl = xbar.p_mirror.replicate(i_lane)  # [mux, ..., 2, n_lane]
        i_io = i_wdl.unflatten(-1, (xbar.n_io, lanes_per_io)).movedim(-1, 0)
        i_dl = xbar.n_mirror.replicate(i_io)  # [lpi, mux, ..., 2, n_io]
        i_sub, sign = xbar.subtractor.subtract(i_dl[..., 0, :], i_dl[..., 1, :], t_conduct__ns=cfg.t_conduct__ns)
        # [lpi, mux, ..., n_io] -> [..., n_io, lpi, mux] -> [..., n_io, io_col_num]
        i_sub = i_sub.movedim(0, -1).movedim(0, -1).flatten(-2)
        sign = sign.movedim(0, -1).movedim(0, -1).flatten(-2)
    return i_sub, sign


def per_phase_clamp_reference(w: Tensor, x: Tensor, *, active_row_num: int, mag_max: int = MAG_MAX) -> Tensor:
    """CPU int64 unit-role reference: per-plane ``clamp(x_p . w_p)`` codes.

    Args:
        w: Ternary digit tensor ``[col_num, 1, row_num]`` (size-1 digit axis).
        x: Binary WL tensor ``[..., row_num]`` (full row; the reference
            derives the sub-phase partials itself).
        active_row_num: Rows per WL sub-phase (plane p owns rows
            ``[p * A, (p + 1) * A)``).
        mag_max: Signed-magnitude clip bound (3-bit -> 7).

    Returns:
        Expected signed per-plane codes ``[..., P, col_num]``
        (``P = row_num / active_row_num``) on CPU (int64; integer matmul
        stays on CPU by design).
    """
    w2 = w.squeeze(-2).cpu().long()  # (col, row)
    x2 = x.cpu().long()
    phase_num = w2.shape[-1] // active_row_num
    xp = x2.unflatten(-1, (phase_num, active_row_num))  # (..., P, A)
    wp = w2.unflatten(-1, (phase_num, active_row_num))  # (col, P, A)
    partial = torch.einsum("...pa,cpa->...pc", xp, wp)  # (..., P, col)
    return partial.clamp(-mag_max, mag_max)


def decode(
    xbar: IsubIadc1t1rCimMacro,
    w: Tensor,
    x: Tensor,
    *,
    adc_operation_point: AdcOperationPoint = ADC_OP,
) -> Tensor:
    """Program ``w``, expand ``x`` into masked WL planes, run one VMM.

    Returns the per-plane codes ``[..., P, col_num]`` on CPU.
    """
    device = tile_device(xbar)
    xbar.program(w.to(device))
    planes = masked_planes(x.to(device), row_num=xbar.config.row_num, max_active_rows=xbar.max_active_rows)
    with torch.no_grad():
        out = xbar.vec_mat_mul(planes, adc_operation_point=adc_operation_point)
    return out.cpu()
