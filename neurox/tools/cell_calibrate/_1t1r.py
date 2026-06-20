"""Calibrate :class:`XbarCell1T1R`'s access-node condensation count.

Single-axis sweep over the per-cell ``n_newton`` (the unrolled Newton on
the access-node KCL ``F_X = I_NMOS(V_X) - I_RRAM(V_X)`` after the Pade
current-divider seed), using **step-ratio plateau detection** (primary)
plus a **relative residual guard** (sanity). Same chip-parameter-free
methodology as :mod:`neurox.tools.solver_calibrate`, applied to the cell's
internal node instead of the array wire / clamp unknowns:

  * Plateau is read from the cell's own ``V_X`` iterate sequence — when
    ``max |V_X,n - V_X,n-1|`` stops shrinking across the operating grid,
    the unrolled Newton has hit fp round-off and further steps do not
    change the answer.

  * Residual guard: ``max |I_NMOS - I_RRAM| / max |I_cell| < reltol``
    (default ``1e-2``), the absolute internal-KCL mismatch returned in
    ``XbarCell1T1RDCOP.residuals.cell__uA`` divided by the grid's own
    ``max |I_cell|`` — workload-derived, not chip-tuned.

The operating grid spans the regime the array solver drives the cell
over: a grid of ``v_bl`` / ``v_sl`` across the read-voltage range, the
word line both off and on, and every programmed RRAM state in
``state_to_g_map__uS``. The plateau is read at the worst (slowest-
converging) point of that grid.

CLI: ``python -m neurox.tools.cell_calibrate._1t1r --help``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.device import NMOSPolicy, RRAMPolicy
from neurox.tools._config import add_standard_args, load_tool_config, setup_logging
from neurox.tools.solver_calibrate._plateau import (
    CandidateRow,
    WorkloadScale,
    pick_with_plateau_and_guard,
)
from neurox.xbar import Offset1T1RXbarConfig
from neurox.xbar._1t1r.cell import (
    XbarCell1T1R,
    XbarCell1T1RConfig,
    XbarCell1T1RPolicy,
    XbarCell1T1RResiduals,
)
from neurox.xbar.cell import XbarCell

# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GridCfg:
    """``[grid]`` section: terminal-voltage / word-line operating sweep.

    Attributes:
        v_terminal_min__V: Minimum bit-line / source-line node voltage [V]
            of the read-voltage sweep.
        v_terminal_max__V: Maximum bit-line / source-line node voltage [V].
        n_terminal: Number of points per terminal axis; the grid is the
            full ``n_terminal x n_terminal`` ``(v_bl, v_sl)`` outer
            product (both rails swept independently).
        v_wl_off__V: Word-line drive [V] for the off state (NMOS cut off).
        v_wl_on__V: Word-line drive [V] for the on state (NMOS conducting).
    """

    v_terminal_min__V: float
    v_terminal_max__V: float
    n_terminal: int
    v_wl_off__V: float
    v_wl_on__V: float


@dataclass(frozen=True)
class _SweepCfg:
    """``[sweep]`` section: candidate counts + plateau / guard knobs."""

    candidates: list[int]
    ratio_threshold: float
    reltol: float
    margin: int


@dataclass(frozen=True)
class _RuntimeCfg:
    """``[runtime]`` section: dtype reproducibility knob.

    The sweep is deterministic over a fixed voltage grid with the
    cell's nonidealities forced off, so there is no RNG; ``dtype``
    is the only numerical knob. fp64 is the calibration default because
    the per-cell condensation is cheap and the round-off floor is the
    quantity being characterised.
    """

    dtype: str


@dataclass(frozen=True)
class CellCalibrate1T1RConfig:
    """Top-level config for :mod:`neurox.tools.cell_calibrate._1t1r`."""

    xbar: Offset1T1RXbarConfig
    grid: _GridCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg


log = logging.getLogger(__name__)


def _all_off_policy() -> XbarCell1T1RPolicy:
    """All-off composite policy so the residual floor is pure round-off."""
    return XbarCell1T1RPolicy(
        rram=RRAMPolicy(
            prog_gamma=False,
            stuck_at=False,
            read_telegraph=False,
            read_thermal=False,
        ),
        nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
    )


def _build_cell(
    cell_config: XbarCell1T1RConfig,
    *,
    n_newton: int,
    device: torch.device,
    dtype: torch.dtype,
) -> XbarCell1T1R:
    """Build a noise-off 1T1R cell with the sweep's condensation count."""
    from dataclasses import replace

    cfg = replace(cell_config, n_newton=n_newton)
    cell = XbarCell.from_config(
        config=cfg,
        policy=_all_off_policy(),
        inst_shape=(1,),
        dtype=dtype,
        T__K=300.0,
    )
    assert isinstance(cell, XbarCell1T1R)
    cell.to(device)
    cell.eval()
    cell.fabricate()
    return cell


def build_operating_grid(
    cell_config: XbarCell1T1RConfig,
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
    cell: XbarCell1T1R,
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
        snapshot = cell.snapshot(control=v_wl_pts, shape=shape, multi_coords=None, t_elapsed=0.0)
        dcop = cell.solve_dc(
            v_bl[mask].reshape(n_pts, 1),
            v_sl[mask].reshape(n_pts, 1),
            snapshot,
            compute_residuals=True,
        )
        residuals = dcop.residuals
        assert isinstance(residuals, XbarCell1T1RResiduals)
        v_x[mask] = dcop.v_x__V.reshape(n_pts)
        cell_residual[mask] = residuals.cell__uA.reshape(n_pts)
        i_cell[mask] = dcop.i__uA.reshape(n_pts)
    return v_x, cell_residual, i_cell


def sweep_n_newton(
    *,
    cell_config: XbarCell1T1RConfig,
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


def _format_row(row: CandidateRow) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    return f"n_newton={row.iter_count:3d}  step_v_x={step}  residual.cell.max={row.residual_max['cell__uA']:9.2e} uA"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate XbarCell1T1R access-node n_newton via step-ratio plateau.")
    add_standard_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(CellCalibrate1T1RConfig, args.config)
    log.info("loaded config from %s", args.config)

    cell_config = cfg.xbar.core_config.cell_config
    if not isinstance(cell_config, XbarCell1T1RConfig):
        raise SystemExit(f"cell config must be XbarCell1T1RConfig; got {type(cell_config).__name__}")
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    n_states = len(cell_config.state_to_g_map__uS)
    n_grid = cfg.grid.n_terminal * cfg.grid.n_terminal * 2 * n_states
    log.info("=" * 80)
    log.info("XbarCell1T1R access-node — step-ratio plateau calibration")
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
    log.info("TOML fragment for chip preset [xbar.core_config.cell_config]:")
    log.info("    n_newton = %d", final)
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
