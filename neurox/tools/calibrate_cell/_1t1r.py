"""Calibrate :class:`XbarCell1t1rDetail`'s access-node condensation count and
emit the two chip-config fragments derived from one run:

  1. the margined ``n_newton`` pick for the Detail cell fragment;
  2. a :class:`XbarCell1t1rLinearConfig` fragment whose per-(state, WL-level)
     chord conductance and BL-side drop fraction reproduce the Detail model's
     branch current and access node at a nominal operating point.

CLI: ``python -m neurox.tools.calibrate_cell._1t1r --help``
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, replace

import tomli_w
import torch
from torch import Tensor

from neurox.common import ConfigBase
from neurox.primitive.device import MosfetPolicy, RramPolicy
from neurox.primitive.xbar.cell import (
    XbarCell1t1rDetail,
    XbarCell1t1rDetailConfig,
    XbarCell1t1rDetailPolicy,
    XbarCell1t1rDetailProber,
    XbarCell1t1rLinearConfig,
)
from neurox.tools._config import add_standard_args, load_tool_config, setup_logging
from neurox.tools._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GridCfg:
    """``[grid]`` section: terminal-voltage / word-line operating sweep.

    Attributes:
        v_terminal_min__V: Minimum bit-line / source-line node voltage
            of the read-voltage sweep.
        v_terminal_max__V: Maximum bit-line / source-line node voltage.
        n_terminal: Number of points per terminal axis; the grid is the
            full ``n_terminal x n_terminal`` ``(v_bl, v_sl)`` outer
            product (both rails swept independently).
        v_wl_off__V: Word-line drive for the off state (NMOS cut off).
        v_wl_on__V: Word-line drive for the on state (NMOS conducting).
        v_bl_op__V: Nominal bit-line operating voltage the linearized
            cell's chord conductance and drop fraction are extracted at.
        v_sl_op__V: Nominal source-line operating voltage for the same
            extraction.
    """

    v_terminal_min__V: float
    v_terminal_max__V: float
    n_terminal: int
    v_wl_off__V: float
    v_wl_on__V: float
    v_bl_op__V: float
    v_sl_op__V: float


@dataclass(frozen=True)
class _SweepCfg:
    """``[sweep]`` section: candidate counts + plateau / guard knobs."""

    candidates: list[int]
    ratio_threshold: float
    reltol: float
    margin: int


@dataclass(frozen=True)
class _RuntimeCfg:
    """``[runtime]`` section: dtype reproducibility knob."""

    dtype: str


class CalibrateCell1t1rConfig(ConfigBase):
    """Top-level config for :mod:`neurox.tools.calibrate_cell._1t1r`.

    ``cell_config`` is the Detail cell fragment under calibration, pulled
    from any scheme's chip params via ``_neurox_use`` (a dotted section name
    reaches a nested ``cell_config`` table).
    """

    cell_config: XbarCell1t1rDetailConfig
    grid: _GridCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg


log = logging.getLogger(__name__)


def _all_off_policy() -> XbarCell1t1rDetailPolicy:
    """All-off composite policy so the residual floor is pure round-off."""
    return XbarCell1t1rDetailPolicy(
        rram_policy=RramPolicy(
            prog_gamma=False,
            stuck_at=False,
            read_telegraph=False,
            read_thermal=False,
        ),
        nmos_policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )


def _build_cell(
    cell_config: XbarCell1t1rDetailConfig,
    *,
    n_newton: int,
    device: torch.device,
    dtype: torch.dtype,
) -> XbarCell1t1rDetail:
    """Build a noise-off Detail 1T1R cell with the given condensation count."""
    cfg = replace(cell_config, n_newton=n_newton)
    cell = XbarCell1t1rDetail(
        config=cfg,
        policy=_all_off_policy(),
        inst_shape=(1,),
        dtype=dtype,
        T__K=300.0,
    )
    cell.to(device)
    cell.eval()
    cell.fabricate()
    return cell


def build_operating_grid(
    cell_config: XbarCell1t1rDetailConfig,
    grid: _GridCfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Enumerate the representative operating grid as flat per-point tensors.

    The grid is the outer product of:

      * ``v_bl`` over the read-voltage range (``n_terminal`` points),
      * ``v_sl`` over the same range (``n_terminal`` points),
      * the word line off and on,
      * every programmed RRAM state in ``state_to_g_map__uS``.

    Returns ``(v_bl, v_sl, v_wl, state_idx)``, each a 1-D tensor of the
    same length ``n_terminal**2 * 2 * n_states``. ``state_idx`` is a
    ``long`` tensor of the programmed-state index at each point; the
    caller programs the cell per distinct state and selects the matching
    points.
    """
    v_axis = torch.linspace(
        grid.v_terminal_min__V,
        grid.v_terminal_max__V,
        grid.n_terminal,
        dtype=dtype,
        device=device,
    )
    v_wl_axis = torch.tensor([grid.v_wl_off__V, grid.v_wl_on__V], dtype=dtype, device=device)
    n_states = len(cell_config.state_to_g_map__uS)
    state_axis = torch.arange(n_states, dtype=torch.long, device=device)

    # Cartesian product over (v_bl, v_sl, v_wl, state) flattened to 1-D.
    grids = torch.meshgrid(v_axis, v_axis, v_wl_axis, state_axis.to(dtype), indexing="ij")
    v_bl = grids[0].reshape(-1)
    v_sl = grids[1].reshape(-1)
    v_wl = grids[2].reshape(-1)
    state_idx = grids[3].reshape(-1).round().long()
    return v_bl, v_sl, v_wl, state_idx


def _solve_grid_for_candidate(
    cell: XbarCell1t1rDetail,
    *,
    v_bl: Tensor,
    v_sl: Tensor,
    v_wl: Tensor,
    state_idx: Tensor,
    n_states: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve every grid point at one candidate cell, grouped by RRAM state.

    The cell programs one RRAM conductance at a time, so the grid is
    solved per distinct programmed state and the per-point ``V_X`` /
    ``cell__uA`` / ``I_cell`` are scattered back into flat tensors aligned
    with the input grid order.

    Returns ``(v_x, cell_residual__uA, i_cell__uA)`` over the full grid.
    """
    v_x = torch.empty_like(v_bl)
    cell_residual = torch.empty_like(v_bl)
    i_cell = torch.empty_like(v_bl)

    for s in range(n_states):
        mask = state_idx == s
        if not bool(mask.any()):
            continue
        # Program the whole cell to this RRAM state, then evaluate the
        # subset of grid points that use it. The cell's ``inst_shape`` is
        # ``(1,)``; we drive a ``(n_pts, 1)`` view so the cell's snapshot /
        # solve broadcast over the ``col``/``row`` trailing pair.
        cell.program(torch.full((1,), s, dtype=torch.long, device=v_bl.device))
        n_pts = int(mask.sum())
        shape = (n_pts, 1)
        v_wl_pts = v_wl[mask].reshape(n_pts, 1)
        snap = cell.snapshot(control=v_wl_pts, shape=shape, multi_coords=None, t_elapsed=0.0)
        with XbarCell1t1rDetailProber() as cp:
            dcop = cell.solve_dc(
                v_bl[mask].reshape(n_pts, 1),
                v_sl[mask].reshape(n_pts, 1),
                snap,
            )
        records = cp.records
        assert len(records) == 1
        v_x[mask] = dcop.v_x__V.reshape(n_pts)
        cell_residual[mask] = records[0].cell__uA.reshape(n_pts)
        i_cell[mask] = dcop.i__uA.reshape(n_pts)
    return v_x, cell_residual, i_cell


def sweep_n_newton(
    *,
    cell_config: XbarCell1t1rDetailConfig,
    candidates: list[int],
    grid: _GridCfg,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[CandidateRow], WorkloadScale]:
    """Run the cell at each candidate ``n_newton`` and collect step + residual.

    The cell has one internal unknown per grid point (``V_X``), so the
    step-delta is a point-wise comparison reduced to its grid maximum. The
    residual is the absolute internal-KCL mismatch ``|I_NMOS - I_RRAM|``
    on the cell DCOP (``residuals.cell__uA``), reduced to its grid maximum.
    """
    v_bl, v_sl, v_wl, state_idx = build_operating_grid(cell_config, grid, device=device, dtype=dtype)
    n_states = len(cell_config.state_to_g_map__uS)

    rows: list[CandidateRow] = []
    i_cell_typ__uA = 0.0
    v_x_prev: Tensor | None = None
    for n_newton in candidates:
        cell = _build_cell(cell_config, n_newton=n_newton, device=device, dtype=dtype)
        v_x, cell_residual, i_cell = _solve_grid_for_candidate(
            cell,
            v_bl=v_bl,
            v_sl=v_sl,
            v_wl=v_wl,
            state_idx=state_idx,
            n_states=n_states,
        )
        residual_max__uA = float(cell_residual.abs().max().item())
        i_cell_typ__uA = max(i_cell_typ__uA, float(i_cell.abs().max().item()))

        if v_x_prev is None:
            step_max__V: float | None = None
        else:
            step_max__V = float((v_x - v_x_prev).abs().max().item())

        rows.append(
            CandidateRow(
                iter_count=n_newton,
                step_max__V=step_max__V,
                step_per_class__V={"v_x": step_max__V if step_max__V is not None else 0.0},
                residual_max={"cell__uA": residual_max__uA},
            )
        )
        v_x_prev = v_x

    # The cell residual is a current; the voltage scale is unused for this
    # workload (no clamp residual). Pass a dummy to satisfy the dataclass.
    scale = WorkloadScale(i_cell_typ__uA=i_cell_typ__uA, v_node_typ__V=1.0)
    return rows, scale


# ---------------------------------------------------------------------------
# Linear-cell fragment extraction
# ---------------------------------------------------------------------------

_VX_RATIO_TOL = 1e-9
"""Tolerance for clamping fp excursions of ``vx_ratio`` just outside [0, 1]."""


def _chord_params(
    i__uA: float,
    v_x__V: float,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
    label: str,
) -> tuple[float, float]:
    """``(g_cell__uS, vx_ratio)`` of one converged Detail branch at the OP.

    Both quantities put the fixed read span ``v_bl_op - v_sl_op`` in the
    denominator, so a cut-off branch stays well-conditioned: its chord
    conductance is its honest (possibly zero) leakage value. ``vx_ratio``
    is clamped into ``[0, 1]`` only for tiny fp excursions (logged); a
    gross excursion or a non-finite value raises.
    """
    span__V = v_bl_op__V - v_sl_op__V
    g_cell__uS = i__uA / span__V
    vx_ratio = (v_bl_op__V - v_x__V) / span__V
    if not (math.isfinite(g_cell__uS) and math.isfinite(vx_ratio)):
        raise ValueError(f"non-finite chord params {label}: g_cell = {g_cell__uS!r} uS, vx_ratio = {vx_ratio!r}")
    if not (-_VX_RATIO_TOL <= vx_ratio <= 1.0 + _VX_RATIO_TOL):
        raise ValueError(f"vx_ratio {label} grossly outside [0, 1]: {vx_ratio!r}")
    clamped = min(max(vx_ratio, 0.0), 1.0)
    if clamped != vx_ratio:
        log.info("clamped vx_ratio %s from %.17g into [0, 1]", label, vx_ratio)
    return g_cell__uS, clamped


def extract_linear_cell_config(
    cell_config: XbarCell1t1rDetailConfig,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
    v_wl_off__V: float,
    v_wl_on__V: float,
    device: torch.device,
    dtype: torch.dtype,
    n_newton: int | None = None,
) -> XbarCell1t1rLinearConfig:
    """Extract the linearized-cell config from the Detail model at one OP.

    Solves the noise-off Detail cell exactly at ``(v_bl_op__V, v_sl_op__V)``
    for every programmed state at both WL levels, and converts each converged
    branch ``(I, V_X)`` into the divider pair

    ``g_cell = I / (v_bl_op - v_sl_op)``,
    ``vx_ratio = (v_bl_op - V_X) / (v_bl_op - v_sl_op)``,

    so the linear branch reproduces the Detail branch current and access
    node at the operating point. The fixed read span in both denominators
    keeps cut-off branches well-conditioned: their chord conductance is the
    honest (possibly zero) leakage value. The four shared node-to-ground
    capacitances copy verbatim from ``cell_config``; the WL on/off
    threshold is the midpoint of the two WL levels.

    Args:
        cell_config: Detail cell fragment under calibration.
        v_bl_op__V: Nominal bit-line operating voltage.
        v_sl_op__V: Nominal source-line operating voltage.
        v_wl_off__V: Word-line off drive.
        v_wl_on__V: Word-line on drive.
        device: Torch device for the solves.
        dtype: Tensor dtype for the solves.
        n_newton: Condensation count override for the extraction solves;
            ``None`` keeps ``cell_config.n_newton``.

    Returns:
        A validated, buildable :class:`XbarCell1t1rLinearConfig`.
    """
    count = cell_config.n_newton if n_newton is None else n_newton
    cell = _build_cell(cell_config, n_newton=count, device=device, dtype=dtype)
    n_states = len(cell_config.state_to_g_map__uS)
    v_bl = torch.full((1, 1), v_bl_op__V, dtype=dtype, device=device)
    v_sl = torch.full((1, 1), v_sl_op__V, dtype=dtype, device=device)

    g_cell_off: list[float] = []
    g_cell_on: list[float] = []
    vx_ratio_off: list[float] = []
    vx_ratio_on: list[float] = []
    for s in range(n_states):
        cell.program(torch.full((1,), s, dtype=torch.long, device=device))
        for level, v_wl__V, g_table, vx_table in (
            ("off", v_wl_off__V, g_cell_off, vx_ratio_off),
            ("on", v_wl_on__V, g_cell_on, vx_ratio_on),
        ):
            v_wl = torch.full((1, 1), v_wl__V, dtype=dtype, device=device)
            snap = cell.snapshot(control=v_wl, shape=(1, 1), multi_coords=None, t_elapsed=0.0)
            dcop = cell.solve_dc(v_bl, v_sl, snap)
            g_cell__uS, vx_ratio = _chord_params(
                float(dcop.i__uA),
                float(dcop.v_x__V),
                v_bl_op__V=v_bl_op__V,
                v_sl_op__V=v_sl_op__V,
                label=f"(state {s}, wl {level})",
            )
            g_table.append(g_cell__uS)
            vx_table.append(vx_ratio)

    return XbarCell1t1rLinearConfig(
        c_bl__fF=cell_config.c_bl__fF,
        c_x__fF=cell_config.c_x__fF,
        c_sl__fF=cell_config.c_sl__fF,
        c_wl__fF=cell_config.c_wl__fF,
        g_cell_off_table__uS=tuple(g_cell_off),
        g_cell_on_table__uS=tuple(g_cell_on),
        vx_ratio_off_table=tuple(vx_ratio_off),
        vx_ratio_on_table=tuple(vx_ratio_on),
        v_wl_on_threshold__V=(v_wl_off__V + v_wl_on__V) / 2.0,
    )


# ---------------------------------------------------------------------------
# Fragment emission
# ---------------------------------------------------------------------------

_FRAGMENT_SECTION = "cell_config"
"""Top-level table both emitted fragments live under."""


def n_newton_fragment_text(n_newton: int) -> str:
    """The Detail ``n_newton`` fragment as TOML text (header comment + table)."""
    header = (
        "# Detail-cell n_newton pick emitted by neurox.tools.calibrate_cell\n"
        "# (step-ratio plateau + margin). Merge into the scheme's Detail cell\n"
        "# fragment (its cell_config table).\n"
    )
    return header + tomli_w.dumps({_FRAGMENT_SECTION: {"n_newton": n_newton}})


def linear_fragment_text(
    linear_config: XbarCell1t1rLinearConfig,
    *,
    v_bl_op__V: float,
    v_sl_op__V: float,
) -> str:
    """The linearized-cell fragment as TOML text (header comment + table)."""
    header = (
        "# Linearized 1T1R cell fragment emitted by neurox.tools.calibrate_cell.\n"
        "# Per-state chord conductance g_cell = I / (v_bl - v_sl) and BL-side\n"
        "# drop fraction vx_ratio = (v_bl - V_X) / (v_bl - v_sl) of the Detail\n"
        "# cell, one flat table per WL level (off / on), at the nominal\n"
        f"# operating point v_bl = {v_bl_op__V} V, v_sl = {v_sl_op__V} V; the WL\n"
        "# threshold is the midpoint of the calibration grid's off/on WL\n"
        "# levels. Selecting it is a pure config choice: point the array's\n"
        "# cell_config table at this file, e.g.\n"
        "#   [cim_macro.array_config.cell_config]\n"
        '#   _neurox_use = "cell_linear.toml:cell_config"\n'
    )
    return header + tomli_w.dumps({_FRAGMENT_SECTION: linear_config.to_dict()})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate XbarCell1t1rDetail access-node n_newton via step-ratio plateau "
        "and emit the Detail / Linear config fragments."
    )
    add_standard_args(parser, output_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(CalibrateCell1t1rConfig, args.config)
    log.info("loaded config from %s", args.config)

    cell_config = cfg.cell_config
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    n_states = len(cell_config.state_to_g_map__uS)
    n_grid = cfg.grid.n_terminal * cfg.grid.n_terminal * 2 * n_states
    log.info("=" * 80)
    log.info("XbarCell1t1rDetail access-node — step-ratio plateau calibration")
    log.info(
        "grid: v_bl/v_sl in [%.3f, %.3f] V x %d^2, WL in {%.3f, %.3f} V, %d states -> %d points",
        cfg.grid.v_terminal_min__V,
        cfg.grid.v_terminal_max__V,
        cfg.grid.n_terminal,
        cfg.grid.v_wl_off__V,
        cfg.grid.v_wl_on__V,
        n_states,
        n_grid,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, margin=%d, dtype=%s",
        cfg.sweep.ratio_threshold,
        cfg.sweep.reltol,
        cfg.sweep.margin,
        cfg.runtime.dtype,
    )
    log.info("=" * 80)

    rows, scale = sweep_n_newton(
        cell_config=cell_config,
        candidates=cfg.sweep.candidates,
        grid=cfg.grid,
        device=device,
        dtype=dtype,
    )

    log.info("grid scale: max|I_cell|=%.3e uA", scale.i_cell_typ__uA)
    log.info("")
    for r in rows:
        log.info(_format_row(r))
    log.info("")

    pick = pick_with_plateau_and_guard(
        rows,
        scale,
        ratio_threshold=cfg.sweep.ratio_threshold,
        reltol=cfg.sweep.reltol,
    )

    if pick.iter_count is None:
        log.error("Calibration failed: %s", pick.reason)
        return 2

    final = pick.iter_count + cfg.sweep.margin
    log.info("=" * 80)
    log.info("Picked n_newton = %d  (%s)", pick.iter_count, pick.reason)
    log.info("Recommended with margin %d: n_newton = %d", cfg.sweep.margin, final)
    log.info(
        "Residual guard ratio at pick: cell=%.3e  (reltol = %.1e)",
        pick.residual_guard_ratios.get("cell__uA", 0.0),
        cfg.sweep.reltol,
    )
    log.info("")
    log.info("TOML fragment for the scheme's Detail cell_config table:")
    log.info("    n_newton = %d", final)
    log.info("=" * 80)

    # --- Linear-cell fragment at the nominal operating point ---
    linear_config = extract_linear_cell_config(
        cell_config,
        v_bl_op__V=cfg.grid.v_bl_op__V,
        v_sl_op__V=cfg.grid.v_sl_op__V,
        v_wl_off__V=cfg.grid.v_wl_off__V,
        v_wl_on__V=cfg.grid.v_wl_on__V,
        device=device,
        dtype=dtype,
        n_newton=final,
    )
    log.info(
        "Linear-cell divider tables at OP (v_bl = %.3f V, v_sl = %.3f V):",
        cfg.grid.v_bl_op__V,
        cfg.grid.v_sl_op__V,
    )
    per_state = zip(
        linear_config.g_cell_off_table__uS,
        linear_config.g_cell_on_table__uS,
        linear_config.vx_ratio_off_table,
        linear_config.vx_ratio_on_table,
        strict=True,
    )
    for s, (g_off, g_on, vx_off, vx_on) in enumerate(per_state):
        log.info(
            "  state %d: g_cell(off/on) = %.6e / %.6e uS, vx_ratio(off/on) = %.9f / %.9f",
            s,
            g_off,
            g_on,
            vx_off,
            vx_on,
        )
    log.info("=" * 80)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        n_newton_path = args.output_dir / "cell_detail_n_newton.toml"
        n_newton_path.write_text(n_newton_fragment_text(final))
        log.info("wrote Detail n_newton fragment to %s", n_newton_path)
        linear_path = args.output_dir / "cell_linear.toml"
        linear_path.write_text(
            linear_fragment_text(linear_config, v_bl_op__V=cfg.grid.v_bl_op__V, v_sl_op__V=cfg.grid.v_sl_op__V)
        )
        log.info("wrote Linear cell fragment to %s", linear_path)
    return 0


def _format_row(row: CandidateRow) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    return f"n_newton={row.iter_count:3d}  step_v_x={step}  residual.cell.max={row.residual_max['cell__uA']:9.2e} uA"


if __name__ == "__main__":
    raise SystemExit(main())
