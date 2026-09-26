"""Extract a linear 1T1R model from settled Detail operating points."""

from __future__ import annotations

import math
from typing import Literal

import tomli_w
import torch

from neurox.common.module import ConfigBase
from neurox.primitive.device.mosfet import MosfetPolicy
from neurox.primitive.device.rram import RramPolicy
from neurox.primitive.xbar.cell import (
    XbarCell1t1rDetail,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rLinearConfig,
)
from neurox.tools.module import prepare_module

__all__ = [
    "CalibrateCellX1t1rConfig",
    "extract_linear_cell_config",
    "linear_fragment_text",
]


class CalibrateCellX1t1rConfig(ConfigBase):
    cell_config: XbarCell1t1rDetailConfig
    dtype: Literal["float32", "float64"]
    v_bl_op__V: float
    v_sl_op__V: float
    v_wl_off__V: float
    v_wl_on__V: float

    def validate(self) -> None:
        self._require_gt(self.v_bl_op__V, "v_bl_op__V", self.v_sl_op__V)
        self._require_gt(self.v_wl_on__V, "v_wl_on__V", self.v_wl_off__V)


_VX_RATIO_TOL = 1e-9


def _detail_policy() -> XbarCell1t1rDetailPolicy:
    """Disable stochastic effects so every candidate sees the same circuit."""
    return XbarCell1t1rDetailPolicy(
        rram_policy=RramPolicy(
            prog_gamma=False,
            drift=False,
            stuck_at=False,
            read_telegraph=False,
            read_thermal=False,
        ),
        nmos_policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )


def _build_cell(
    cell_config: XbarCell1t1rDetailConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> XbarCell1t1rDetail:
    """Build one deterministic canonical Detail cell."""
    cell = XbarCell1t1rDetail(
        config=cell_config,
        policy=_detail_policy(),
        inst_shape=(1,),
        dtype=dtype,
    )
    return prepare_module(cell, device=device)


def _chord_params(
    i__uA: float,
    v_x__V: float,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
    label: str,
) -> tuple[float, float]:
    """Return chord conductance and BL-side drop fraction at one point."""
    span__V = v_bl_op__V - v_sl_op__V
    if span__V <= 0.0:
        raise ValueError("linear extraction requires v_bl_op__V > v_sl_op__V")
    g_cell__uS = i__uA / span__V
    vx_ratio = (v_bl_op__V - v_x__V) / span__V
    if not (math.isfinite(g_cell__uS) and math.isfinite(vx_ratio)):
        raise ValueError(f"non-finite chord params {label}: g_cell__uS={g_cell__uS!r}, vx_ratio={vx_ratio!r}")
    if g_cell__uS < 0.0:
        raise ValueError(f"negative chord conductance {label}: {g_cell__uS!r} uS")
    if not (-_VX_RATIO_TOL <= vx_ratio <= 1.0 + _VX_RATIO_TOL):
        raise ValueError(f"vx_ratio {label} grossly outside [0, 1]: {vx_ratio!r}")
    return g_cell__uS, min(max(vx_ratio, 0.0), 1.0)


def _solve_linear_point(
    cell: XbarCell1t1rDetail,
    *,
    state: int,
    v_bl_op__V: float,
    v_sl_op__V: float,
    v_wl__V: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Solve one deterministic Detail operating point."""
    cell.program(torch.full((1,), state, dtype=torch.int32, device=device))
    v_bl__V = torch.full((1, 1), v_bl_op__V, dtype=dtype, device=device)
    v_sl__V = torch.full((1, 1), v_sl_op__V, dtype=dtype, device=device)
    v_wl_grid__V = torch.full((1, 1), v_wl__V, dtype=dtype, device=device)
    cell_snap = cell.snapshot(control=v_wl_grid__V, shape=(1, 1))
    cell_dcop = cell.solve_dc(v_bl__V=v_bl__V, v_sl__V=v_sl__V, snap=cell_snap)
    return float(cell_dcop.i__uA), float(cell_dcop.v_x__V)


def extract_linear_cell_config(
    cell_config: XbarCell1t1rDetailConfig,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
    v_wl_off__V: float,
    v_wl_on__V: float,
    device: torch.device,
    dtype: torch.dtype,
) -> XbarCell1t1rLinearConfig:
    """Calibrate on/off divider tables at one declared terminal operating point.

    Build a new detailed cell on `device` in `dtype`, disable its stochastic
    policies, and solve every programmed state at both word-line levels. The
    supplied config is unchanged. Returned tables preserve state order; their
    switching threshold is the midpoint of the two supplied word-line voltages.
    Use float32 or float64 and `v_bl_op__V > v_sl_op__V`, with the on level
    above the off level. Scalar result extraction can synchronize the device.

    Args:
        cell_config: Detailed-cell model whose weight states are calibrated.
        v_bl_op__V: Bit-line calibration voltage, greater than the source-line
            voltage.
        v_sl_op__V: Source-line calibration voltage.
        v_wl_off__V: Word-line voltage for the off-state tables.
        v_wl_on__V: Word-line voltage for the on-state tables, above the off
            level.
        device: Device for deterministic electrical solves.
        dtype: Float32 or float64 computation dtype for extraction.

    Returns:
        Linear-cell configuration fitted at the supplied operating point. This
        does not validate the approximation at other voltages or temperatures.

    Raises:
        ValueError: The operating span or derived table values are invalid.
        RuntimeError: The detailed electrical solve fails its convergence
            checks.
    """
    cell = _build_cell(cell_config, device=device, dtype=dtype)
    g_cell_off: list[float] = []
    g_cell_on: list[float] = []
    vx_ratio_off: list[float] = []
    vx_ratio_on: list[float] = []
    for state in range(len(cell_config.state_to_g_map__uS)):
        for level, v_wl__V, g_table, vx_table in (
            ("off", v_wl_off__V, g_cell_off, vx_ratio_off),
            ("on", v_wl_on__V, g_cell_on, vx_ratio_on),
        ):
            i__uA, v_x__V = _solve_linear_point(
                cell,
                state=state,
                v_bl_op__V=v_bl_op__V,
                v_sl_op__V=v_sl_op__V,
                v_wl__V=v_wl__V,
                device=device,
                dtype=dtype,
            )
            g_cell__uS, vx_ratio = _chord_params(
                i__uA,
                v_x__V,
                v_bl_op__V=v_bl_op__V,
                v_sl_op__V=v_sl_op__V,
                label=f"(state {state}, wl {level})",
            )
            g_table.append(g_cell__uS)
            vx_table.append(vx_ratio)
    return XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=tuple(g_cell_off),
        g_cell_on_table__uS=tuple(g_cell_on),
        vx_ratio_off_table=tuple(vx_ratio_off),
        vx_ratio_on_table=tuple(vx_ratio_on),
        v_wl_on_threshold__V=(v_wl_off__V + v_wl_on__V) / 2.0,
    )


def linear_fragment_text(
    linear_config: XbarCell1t1rLinearConfig,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
) -> str:
    """Serialize a calibrated linear-cell config as a mergeable TOML fragment.

    The returned text contains a `cell_config` table and comments identifying
    the supplied BL/SL extraction voltages. It performs no solve or file I/O.
    Pass the same voltages used during extraction so the recorded provenance
    remains true.

    Args:
        linear_config: Extracted linear-cell tables to serialize.
        v_bl_op__V: Bit-line voltage used when those tables were extracted.
        v_sl_op__V: Source-line voltage used when those tables were extracted.

    Returns:
        TOML text containing the cell_config table and operating-point
        provenance.
    """
    header = (
        "# Linearized 1T1R cell fragment emitted by neurox.tools.calibration.cell.\n"
        "# Tables reproduce the canonical Detail cell at\n"
        f"# v_bl_op__V={v_bl_op__V:.17g}, v_sl_op__V={v_sl_op__V:.17g}.\n"
    )
    return header + tomli_w.dumps({"cell_config": linear_config.to_dict()})
