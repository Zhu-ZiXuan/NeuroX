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
from pathlib import Path

import torch

from neurox.xbar._1t1r import (
    FullJacobianSolver1T1RConfig,
    NestedSolver1T1RConfig,
    Solver1T1R,
)

from ._common import aggregate_xbar_sweep, build_xbar_for_calibration
from ._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

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
        ls="--", color="gray", lw=0.7,
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
        ls="--", color="gray", lw=0.7,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate FullJacobianSolver1T1R n_newton via step-ratio plateau.",
    )
    parser.add_argument("--xbar-config", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, default=None)
    parser.add_argument("--inst-shape", type=int, nargs="*", default=[256])
    parser.add_argument("--weight-samples", type=int, default=256)
    parser.add_argument("--input-samples-per-weight", type=int, default=8)
    parser.add_argument("--batch-w", type=int, default=256)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Continuous scan (default: 1..10).",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.5,
        help="Plateau criterion: smallest n* where step_{n*+1}/step_{n*} > THIS. "
        "Smaller = more permissive (plateau declared earlier); larger = stricter.",
    )
    parser.add_argument(
        "--reltol",
        type=float,
        default=1e-2,
        help="Residual safety guard: |residual| / |signal| must be < THIS at the pick. "
        "Workload-derived signal scales (max|I_cell|, max|V_node|).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=1,
        help="Add this many iterations to the picked n_newton for safety.",
    )
    parser.add_argument("--tia-n-newton", type=int, default=3)
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    if not args.xbar_config.is_file():
        raise SystemExit(f"--xbar-config: file not found: {args.xbar_config}")

    inst_shape = tuple(args.inst_shape)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    log.info("=" * 80)
    log.info("FullJacobianSolver — step-ratio plateau calibration")
    log.info(
        "workload: inst=%s, %d weights × %d inputs (batch_w=%d), tia_n_newton=%d",
        inst_shape,
        args.weight_samples,
        args.input_samples_per_weight,
        args.batch_w,
        args.tia_n_newton,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, margin=%d",
        args.ratio_threshold,
        args.reltol,
        args.margin,
    )
    log.info("=" * 80)

    # Build a host xbar; the solver gets swapped per-candidate.
    stub = NestedSolver1T1RConfig(n_outer=1, n_inner=1)
    xbar = build_xbar_for_calibration(
        args.xbar_config,
        device=args.device,
        inst_shape=inst_shape,
        dtype=dtype,
        solver_config=stub,
        tia_n_newton=args.tia_n_newton,
    )

    candidate_solvers = []
    for n in args.candidates:
        cfg = FullJacobianSolver1T1RConfig(n_newton=n)
        solver = Solver1T1R.from_config(
            config=cfg,
            rram=xbar.core.rram,
            nmos=xbar.core.nmos,
            bl_driver=xbar.core.tia,
            sl_driver=xbar.core.sl_driver,
        )
        candidate_solvers.append((n, solver))

    rows, scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=candidate_solvers,
        n_weight=args.weight_samples,
        n_input_per_weight=args.input_samples_per_weight,
        batch_w=args.batch_w,
        distribution_path=args.distribution,
        device=args.device,
        seed=args.seed,
    )

    log.info("workload scale: max|I_cell|=%.3e μA, max|V_BL_node|=%.3e V", scale.i_cell_typ__uA, scale.v_node_typ__V)
    log.info("")
    for r in rows:
        log.info(_format_row(r))
    log.info("")

    pick = pick_with_plateau_and_guard(
        rows,
        scale,
        ratio_threshold=args.ratio_threshold,
        reltol=args.reltol,
    )

    if pick.iter_count is None:
        log.error("Calibration failed: %s", pick.reason)
        log.error("residual guard ratios at attempted pick: %s", pick.residual_guard_ratios)
        raise SystemExit(2)

    final = pick.iter_count + args.margin
    log.info("=" * 80)
    log.info("Picked n_newton = %d  (%s)", pick.iter_count, pick.reason)
    log.info("Recommended with margin %d: n_newton = %d", args.margin, final)
    log.info("Residual guard ratios at pick:")
    for k, v in pick.residual_guard_ratios.items():
        log.info("    %-12s  %.3e  (reltol = %.1e)", k, v, args.reltol)
    log.info("")
    log.info("TOML fragment for chip preset:")
    log.info("    [xbar.core_config.solver_config]")
    log.info('    _neurox_type = "FullJacobianSolver1T1RConfig"')
    log.info("    n_newton = %d", final)
    log.info("=" * 80)

    if args.plot_dir is not None:
        plot_sweep(rows, scale, out_path=args.plot_dir / "full_jacobian_sweep.png", reltol=args.reltol)
        log.info("Plot written to %s", args.plot_dir)


if __name__ == "__main__":
    main()
