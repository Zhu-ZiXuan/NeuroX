"""Calibrate :class:`NestedSolver1T1R`'s ``(n_outer, n_inner)`` pair.

Two-axis sweep using **step-ratio plateau detection** (primary) +
**relative residual guard** (sanity), staged:

  Stage A: fix ``n_inner = n_inner_ref`` (generous), sweep ``n_outer``
           → pick the smallest ``n_outer`` at the step plateau.

  Stage B: fix ``n_outer = pick_outer``, sweep ``n_inner``
           → pick the smallest ``n_inner`` at the step plateau.

Both criteria are chip-parameter-free; see :mod:`._plateau` for details.

CLI: ``python -m neurox.tools.solver_calibrate.nested --help``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from neurox.xbar import Offset1T1RXbar
from neurox.xbar._1t1r import NestedSolver1T1RConfig, Solver1T1R

from ._common import aggregate_xbar_sweep, build_xbar_for_calibration
from ._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

log = logging.getLogger(__name__)


def _format_row(row: CandidateRow, label: str) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    return (
        f"{label}={row.iter_count:3d}  "
        f"step={step}  "
        f"|F|.cell={row.residual_max['cell__uA']:9.2e}  "
        f"wire_bl={row.residual_max['wire_bl__uA']:9.2e}  "
        f"clamp_bl={row.residual_max['clamp_bl__V']:9.2e}"
    )


def _build_candidates(
    xbar: Offset1T1RXbar,
    *,
    axis: str,
    candidates: list[int],
    other_value: int,
) -> list[tuple[int, Solver1T1R]]:
    """Build per-axis sweep candidates with the orthogonal axis pinned."""
    out: list[tuple[int, Solver1T1R]] = []
    for n in candidates:
        if axis == "n_outer":
            cfg = NestedSolver1T1RConfig(n_outer=n, n_inner=other_value)
        elif axis == "n_inner":
            cfg = NestedSolver1T1RConfig(n_outer=other_value, n_inner=n)
        else:
            raise ValueError(axis)
        solver = Solver1T1R.from_config(
            config=cfg,
            rram=xbar.core.rram,
            nmos=xbar.core.nmos,
            bl_driver=xbar.core.tia,
            sl_driver=xbar.core.sl_driver,
        )
        out.append((n, solver))
    return out


def plot_stage(
    rows: list[CandidateRow],
    scale: WorkloadScale,
    *,
    axis_label: str,
    out_path: Path,
    reltol: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs_all = [r.iter_count for r in rows]
    # Three panels: solution step, current residuals (cell/wire), voltage
    # residuals (clamp). Current and voltage residuals are plotted
    # separately so each carries its own guard line.
    fig, (ax_step, ax_curr, ax_volt) = plt.subplots(1, 3, figsize=(15, 4))

    steps = [r.step_max__V for r in rows if r.step_max__V is not None]
    step_xs = [r.iter_count for r in rows if r.step_max__V is not None]
    ax_step.plot(step_xs, steps, marker="o")
    ax_step.set_yscale("log")
    ax_step.set_xlabel(axis_label)
    ax_step.set_ylabel("max |u_n − u_{n-1}| [V]")
    ax_step.set_title(f"Solution step ({axis_label} sweep)")
    ax_step.grid(True, which="both", ls=":", lw=0.4)

    for key, color in (("cell__uA", "C0"), ("wire_bl__uA", "C1"), ("wire_sl__uA", "C2")):
        ax_curr.plot(xs_all, [r.residual_max[key] for r in rows], marker=".", color=color, label=key)
    ax_curr.axhline(
        reltol * scale.i_cell_typ__uA,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} × max|I_cell| = {reltol * scale.i_cell_typ__uA:.2e} μA)",
    )
    ax_curr.set_yscale("log")
    ax_curr.set_xlabel(axis_label)
    ax_curr.set_ylabel("max |residual| [μA]")
    ax_curr.set_title("Current residuals")
    ax_curr.grid(True, which="both", ls=":", lw=0.4)
    ax_curr.legend(fontsize="x-small", loc="upper right")

    for key, color in (("clamp_bl__V", "C3"), ("clamp_sl__V", "C4")):
        ax_volt.plot(xs_all, [r.residual_max[key] for r in rows], marker=".", color=color, label=key)
    ax_volt.axhline(
        reltol * scale.v_node_typ__V,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} × max|V_node| = {reltol * scale.v_node_typ__V:.2e} V)",
    )
    ax_volt.set_yscale("log")
    ax_volt.set_xlabel(axis_label)
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
        description="Calibrate NestedSolver1T1R (n_outer, n_inner) via step-ratio plateau."
    )
    parser.add_argument("--xbar-config", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, default=None)
    parser.add_argument("--inst-shape", type=int, nargs="*", default=[256])
    parser.add_argument("--weight-samples", type=int, default=256)
    parser.add_argument("--input-samples-per-weight", type=int, default=8)
    parser.add_argument("--batch-w", type=int, default=256)
    parser.add_argument(
        "--outer-candidates",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Continuous scan for n_outer (default: 1..10).",
    )
    parser.add_argument(
        "--inner-candidates",
        type=int,
        nargs="+",
        default=list(range(1, 6)),
        help="Continuous scan for n_inner (default: 1..5).",
    )
    parser.add_argument(
        "--inner-ref",
        type=int,
        default=5,
        help="Generous n_inner used during the n_outer sweep (Stage A).",
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
        help="Residual safety guard relative to workload-derived signal scale.",
    )
    parser.add_argument("--outer-margin", type=int, default=1)
    parser.add_argument("--inner-margin", type=int, default=0)
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
    log.info("NestedSolver — step-ratio plateau calibration (2-axis staged)")
    log.info(
        "workload: inst=%s, %d weights × %d inputs (batch_w=%d); TIA n_newton read from preset",
        inst_shape,
        args.weight_samples,
        args.input_samples_per_weight,
        args.batch_w,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, outer_margin=%d, inner_margin=%d",
        args.ratio_threshold,
        args.reltol,
        args.outer_margin,
        args.inner_margin,
    )
    log.info("=" * 80)

    # Build host xbar; solvers swap per candidate.
    stub = NestedSolver1T1RConfig(n_outer=1, n_inner=1)
    xbar = build_xbar_for_calibration(
        args.xbar_config,
        device=args.device,
        inst_shape=inst_shape,
        dtype=dtype,
        solver_config=stub,
    )

    # --- Stage A: sweep n_outer at n_inner = inner_ref ---
    log.info("Stage A: sweep n_outer with n_inner pinned at %d", args.inner_ref)
    log.info("-" * 80)
    outer_candidates = _build_candidates(
        xbar,
        axis="n_outer",
        candidates=args.outer_candidates,
        other_value=args.inner_ref,
    )
    outer_rows, outer_scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=outer_candidates,
        n_weight=args.weight_samples,
        n_input_per_weight=args.input_samples_per_weight,
        batch_w=args.batch_w,
        distribution_path=args.distribution,
        device=args.device,
        seed=args.seed,
    )
    log.info(
        "workload scale: max|I_cell|=%.3e μA, max|V_BL_node|=%.3e V",
        outer_scale.i_cell_typ__uA,
        outer_scale.v_node_typ__V,
    )
    log.info("")
    for r in outer_rows:
        log.info(_format_row(r, label="n_outer"))
    log.info("")
    outer_pick = pick_with_plateau_and_guard(
        outer_rows,
        outer_scale,
        ratio_threshold=args.ratio_threshold,
        reltol=args.reltol,
    )
    if outer_pick.iter_count is None:
        log.error("Stage A failed: %s", outer_pick.reason)
        raise SystemExit(2)
    pick_outer = outer_pick.iter_count
    log.info("Stage A pick: n_outer = %d  (%s)", pick_outer, outer_pick.reason)
    log.info("")
    if args.plot_dir is not None:
        plot_stage(
            outer_rows,
            outer_scale,
            axis_label="n_outer",
            out_path=args.plot_dir / "nested_stageA_outer.png",
            reltol=args.reltol,
        )

    # --- Stage B: sweep n_inner at n_outer = pick_outer ---
    log.info("=" * 80)
    log.info("Stage B: sweep n_inner with n_outer pinned at %d", pick_outer)
    log.info("-" * 80)
    inner_candidates = _build_candidates(
        xbar,
        axis="n_inner",
        candidates=args.inner_candidates,
        other_value=pick_outer,
    )
    inner_rows, inner_scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=inner_candidates,
        n_weight=args.weight_samples,
        n_input_per_weight=args.input_samples_per_weight,
        batch_w=args.batch_w,
        distribution_path=args.distribution,
        device=args.device,
        seed=args.seed,
    )
    log.info("")
    for r in inner_rows:
        log.info(_format_row(r, label="n_inner"))
    log.info("")
    inner_pick = pick_with_plateau_and_guard(
        inner_rows,
        inner_scale,
        ratio_threshold=args.ratio_threshold,
        reltol=args.reltol,
    )
    if inner_pick.iter_count is None:
        # n_inner can plausibly plateau at the smallest candidate — every iter
        # changes u so geometrically that the sweep's 2-point ratio test
        # cannot resolve the descent. Fall back to the smallest candidate
        # but re-verify the residual guard AT THAT candidate (Stage A's guard
        # was at n_inner_ref, which is generous; the smallest n_inner might
        # not meet residuals on its own).
        from ._plateau import check_residual_relative_guard

        fallback = args.inner_candidates[0]
        fallback_row = next(r for r in inner_rows if r.iter_count == fallback)
        passed, ratios = check_residual_relative_guard(fallback_row, inner_scale, reltol=args.reltol)
        if not passed:
            worst = max(ratios.items(), key=lambda kv: kv[1])
            log.error(
                "Stage B plateau not detected AND fallback n_inner=%d fails residual guard: "
                "%s ratio=%.3e >= reltol=%.1e. Widen --inner-candidates or raise --reltol.",
                fallback,
                worst[0],
                worst[1],
                args.reltol,
            )
            raise SystemExit(2)
        log.warning(
            "Stage B plateau not detected — falling back to smallest candidate n_inner=%d "
            "(residual guard re-verified there). Reason: %s",
            fallback,
            inner_pick.reason,
        )
        pick_inner = fallback
    else:
        pick_inner = inner_pick.iter_count
        log.info("Stage B pick: n_inner = %d  (%s)", pick_inner, inner_pick.reason)
    log.info("")
    if args.plot_dir is not None:
        plot_stage(
            inner_rows,
            inner_scale,
            axis_label="n_inner",
            out_path=args.plot_dir / "nested_stageB_inner.png",
            reltol=args.reltol,
        )

    final_outer = pick_outer + args.outer_margin
    final_inner = pick_inner + args.inner_margin
    log.info("=" * 80)
    log.info("Picked pair: (n_outer=%d, n_inner=%d)", pick_outer, pick_inner)
    log.info("Recommended with margins: (n_outer=%d, n_inner=%d)", final_outer, final_inner)
    log.info("Residual guard ratios at outer pick:")
    for k, v in outer_pick.residual_guard_ratios.items():
        log.info("    %-12s  %.3e  (reltol = %.1e)", k, v, args.reltol)
    log.info("")
    log.info("TOML fragment for chip preset:")
    log.info("    [xbar.core_config.solver_config]")
    log.info('    _neurox_type = "NestedSolver1T1RConfig"')
    log.info("    n_outer = %d", final_outer)
    log.info("    n_inner = %d", final_inner)
    log.info("=" * 80)


if __name__ == "__main__":
    main()
