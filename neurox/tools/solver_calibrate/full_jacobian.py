"""Calibrate :class:`FullJacobianSolver1T1R`'s Newton iteration count.

Single-axis sweep over ``n_newton`` using **step-ratio plateau detection**
(primary) plus a **relative residual guard** (sanity). Both criteria are
chip-parameter-free:

  * The plateau is read from the solver's own iterate sequence — when
    ``|u_n − u_{n-1}|`` stops shrinking, the solver has reached its
    numerical floor and further iterations don't change the answer.

  * The residual guard checks ``|F(u)| / |signal|`` against a single
    methodological ``reltol`` (default ``1e-2``, sized for the fp32
    accumulated round-off floor), where ``|signal|`` comes from the
    workload itself (``max|I_cell|`` for currents, ``max|V_BL_node|``
    for voltages). No absolute physical target.

CLI: ``python -m neurox.tools.solver_calibrate.full_jacobian --help``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.tools._config import (
    add_standard_args,
    load_tool_config,
    resolve_relative_path,
    setup_logging,
)
from neurox.xbar import Offset1T1RXbarConfig
from neurox.xbar._1t1r import (
    FullJacobianSolver1T1RConfig,
    NestedSolver1T1RConfig,
    Solver1T1R,
)

from ._common import aggregate_xbar_sweep, build_xbar_for_calibration
from ._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkloadCfg:
    inst_shape: list[int]
    weight_samples: int
    input_samples_per_weight: int
    batch_w: int
    distribution: Path | None = None


@dataclass(frozen=True)
class _SweepCfg:
    candidates: list[int]
    ratio_threshold: float
    reltol: float
    margin: int


@dataclass(frozen=True)
class _RuntimeCfg:
    dtype: str
    seed: int


@dataclass(frozen=True)
class SolverCalibrateFullJacobianConfig:
    """Top-level config for :mod:`neurox.tools.solver_calibrate.full_jacobian`."""

    xbar: Offset1T1RXbarConfig
    workload: _WorkloadCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg


log = logging.getLogger(__name__)


def _format_row(row: CandidateRow) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    return (
        f"n_newton={row.iter_count:3d}  "
        f"step={step}  "
        f"|F|.cell={row.residual_max['cell__uA']:9.2e}  "
        f"wire_bl={row.residual_max['wire_bl__uA']:9.2e}  "
        f"clamp_bl={row.residual_max['clamp_bl__V']:9.2e}"
    )


def plot_sweep(
    rows: list[CandidateRow],
    scale: WorkloadScale,
    *,
    out_path: Path,
    reltol: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.iter_count for r in rows]
    # Three panels: solution step, current residuals (cell/wire), voltage
    # residuals (clamp). Current and voltage residuals are plotted
    # separately so each carries its own guard line (current vs voltage
    # signal scales differ in both magnitude and units).
    fig, (ax_step, ax_curr, ax_volt) = plt.subplots(1, 3, figsize=(15, 4))

    steps = [r.step_max__V for r in rows if r.step_max__V is not None]
    step_xs = [r.iter_count for r in rows if r.step_max__V is not None]
    ax_step.plot(step_xs, steps, marker="o", label="max |u_n − u_{n-1}|")
    ax_step.set_yscale("log")
    ax_step.set_xlabel("n_newton")
    ax_step.set_ylabel("step magnitude [V]")
    ax_step.set_title("Solution step (plateau ≡ convergence)")
    ax_step.grid(True, which="both", ls=":", lw=0.4)
    ax_step.legend(fontsize="small")

    for key, color in (("cell__uA", "C0"), ("wire_bl__uA", "C1"), ("wire_sl__uA", "C2")):
        ax_curr.plot(xs, [r.residual_max[key] for r in rows], marker=".", color=color, label=key)
    ax_curr.axhline(
        reltol * scale.i_cell_typ__uA,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} × max|I_cell| = {reltol * scale.i_cell_typ__uA:.2e} μA)",
    )
    ax_curr.set_yscale("log")
    ax_curr.set_xlabel("n_newton")
    ax_curr.set_ylabel("max |residual| [μA]")
    ax_curr.set_title("Current residuals")
    ax_curr.grid(True, which="both", ls=":", lw=0.4)
    ax_curr.legend(fontsize="x-small", loc="upper right")

    for key, color in (("clamp_bl__V", "C3"), ("clamp_sl__V", "C4")):
        ax_volt.plot(xs, [r.residual_max[key] for r in rows], marker=".", color=color, label=key)
    ax_volt.axhline(
        reltol * scale.v_node_typ__V,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} × max|V_node| = {reltol * scale.v_node_typ__V:.2e} V)",
    )
    ax_volt.set_yscale("log")
    ax_volt.set_xlabel("n_newton")
    ax_volt.set_ylabel("max |residual| [V]")
    ax_volt.set_title("Voltage residuals")
    ax_volt.grid(True, which="both", ls=":", lw=0.4)
    ax_volt.legend(fontsize="x-small", loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate FullJacobianSolver1T1R n_newton via step-ratio plateau.",
    )
    add_standard_args(parser, plot_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(SolverCalibrateFullJacobianConfig, args.config)
    log.info("loaded config from %s", args.config)

    inst_shape = tuple(cfg.workload.inst_shape)
    if len(inst_shape) != 1 or inst_shape[0] != cfg.workload.batch_w:
        raise SystemExit(
            f"[workload].inst_shape ({list(inst_shape)}) must be exactly "
            f"[batch_w]={[cfg.workload.batch_w]} — the two axes are bound by the "
            "xbar's program(w) shape contract."
        )
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    distribution_path = resolve_relative_path(cfg.workload.distribution, args.config)

    log.info("=" * 80)
    log.info("FullJacobianSolver — step-ratio plateau calibration")
    log.info(
        "workload: inst=%s, %d weights × %d inputs (batch_w=%d); TIA n_newton read from preset",
        inst_shape,
        cfg.workload.weight_samples,
        cfg.workload.input_samples_per_weight,
        cfg.workload.batch_w,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, margin=%d",
        cfg.sweep.ratio_threshold,
        cfg.sweep.reltol,
        cfg.sweep.margin,
    )
    log.info("=" * 80)

    stub = NestedSolver1T1RConfig(n_outer=1, n_inner=1)
    xbar = build_xbar_for_calibration(
        args.config,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
        solver_config=stub,
    )

    candidate_solvers = []
    for n in cfg.sweep.candidates:
        solver_cfg = FullJacobianSolver1T1RConfig(n_newton=n)
        solver = Solver1T1R.from_config(
            config=solver_cfg,
            rram=xbar.core.rram,
            nmos=xbar.core.nmos,
            bl_driver=xbar.core.tia,
            sl_driver=xbar.core.sl_driver,
        )
        candidate_solvers.append((n, solver))

    rows, scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=candidate_solvers,
        n_weight=cfg.workload.weight_samples,
        n_input_per_weight=cfg.workload.input_samples_per_weight,
        batch_w=cfg.workload.batch_w,
        distribution_path=distribution_path,
        device=device,
        seed=cfg.runtime.seed,
    )

    log.info("workload scale: max|I_cell|=%.3e μA, max|V_BL_node|=%.3e V", scale.i_cell_typ__uA, scale.v_node_typ__V)
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
        log.error("residual guard ratios at attempted pick: %s", pick.residual_guard_ratios)
        return 2

    final = pick.iter_count + cfg.sweep.margin
    log.info("=" * 80)
    log.info("Picked n_newton = %d  (%s)", pick.iter_count, pick.reason)
    log.info("Recommended with margin %d: n_newton = %d", cfg.sweep.margin, final)
    log.info("Residual guard ratios at pick:")
    for k, v in pick.residual_guard_ratios.items():
        log.info("    %-12s  %.3e  (reltol = %.1e)", k, v, cfg.sweep.reltol)
    log.info("")
    log.info("TOML fragment for chip preset:")
    log.info("    [xbar.core_config.solver_config]")
    log.info('    _neurox_type = "FullJacobianSolver1T1RConfig"')
    log.info("    n_newton = %d", final)
    log.info("=" * 80)

    if args.plot_dir is not None:
        plot_sweep(rows, scale, out_path=args.plot_dir / "full_jacobian_sweep.png", reltol=cfg.sweep.reltol)
        log.info("Plot written to %s", args.plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
