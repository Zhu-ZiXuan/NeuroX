"""Calibrate :class:`NestedParallelRailSolver`'s ``(n_outer, n_inner)`` pair.

Two-axis sweep using **step-ratio plateau detection** (primary) +
**relative residual guard** (sanity), staged:

  Stage A: fix ``n_inner = n_inner_ref`` (generous), sweep ``n_outer``
           → pick the smallest ``n_outer`` at the step plateau.

  Stage B: fix ``n_outer = pick_outer``, sweep ``n_inner``
           → pick the smallest ``n_inner`` at the step plateau.

Both criteria are chip-parameter-free; see :mod:`neurox.tools._plateau` for
details.

CLI: ``python -m neurox.tools.calibrate_solver.nested --help``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

# Registers the scheme classes so the tool config's `_neurox_class`
# discriminators can resolve works-defined macro config / policy subclasses.
import neurox.works  # noqa: F401
from neurox.common import ConfigBase
from neurox.primitive.macro.cim import CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig, Solver
from neurox.tools._config import (
    add_standard_args,
    load_tool_config,
    resolve_relative_path,
    setup_logging,
)
from neurox.tools._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

from ._common import aggregate_xbar_sweep, build_xbar_for_calibration

# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkloadCfg:
    """``[workload]`` section: sampling sweep dimensions + optional distribution."""

    inst_shape: list[int]
    weight_samples: int
    input_samples_per_weight: int
    batch_w: int
    distribution: Path | None = None


@dataclass(frozen=True)
class _SweepCfg:
    """``[sweep]`` section: 2-axis candidate iteration counts + criteria."""

    outer_candidates: list[int]
    inner_candidates: list[int]
    inner_ref: int
    ratio_threshold: float
    reltol: float
    outer_margin: int
    inner_margin: int


@dataclass(frozen=True)
class _RuntimeCfg:
    """``[runtime]`` section: dtype + RNG seed + array chunking.

    ``solve_chunk_size`` is the array's solve-chunking knob for the
    calibration runs (a runtime numerical setting, not a physical
    parameter); ``0`` disables chunking (single-block solve).
    """

    dtype: str
    seed: int
    solve_chunk_size: int = 0


@dataclass(frozen=True)
class CalibrateSolverNestedConfig(ConfigBase):
    """Top-level config for :mod:`neurox.tools.calibrate_solver.nested`.

    ``cim_macro`` / ``cim_macro_policy`` are abstract-typed: the TOML selects
    the concrete scheme classes via ``_neurox_class`` (usually by
    ``_neurox_use``-ing a scheme's chip params + all-off policy preset); this
    module imports :mod:`neurox.works` so the discriminators resolve.
    """

    cim_macro: CimMacroConfig
    cim_macro_policy: CimMacroPolicy
    workload: _WorkloadCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg


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
    *,
    axis: str,
    candidates: list[int],
    other_value: int,
) -> list[tuple[int, Solver]]:
    """Build per-axis sweep candidates with the orthogonal axis pinned."""
    out: list[tuple[int, Solver]] = []
    for n in candidates:
        if axis == "n_outer":
            cfg = NestedParallelRailSolverConfig(n_outer=n, n_inner=other_value)
        elif axis == "n_inner":
            cfg = NestedParallelRailSolverConfig(n_outer=other_value, n_inner=n)
        else:
            raise ValueError(axis)
        out.append((n, Solver.from_config(config=cfg)))
    return out


def plot_stage(
    rows: list[CandidateRow],
    scale: WorkloadScale,
    *,
    axis_label: str,
    out_path: Path,
    reltol: float,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate NestedParallelRailSolver (n_outer, n_inner) via step-ratio plateau."
    )
    add_standard_args(parser, plot_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(CalibrateSolverNestedConfig, args.config)
    log.info("loaded config from %s", args.config)

    inst_shape = tuple(cfg.workload.inst_shape)
    # inst_shape[0] is the xbar's parallel weight-program axis; sample_w
    # produces tensors with that exact leading dim. Any mismatch with
    # batch_w trips xbar.program(w)'s shape check immediately.
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
    log.info("NestedParallelRailSolver — step-ratio plateau calibration (2-axis staged)")
    log.info(
        "workload: inst=%s, %d weights × %d inputs (batch_w=%d)",
        inst_shape,
        cfg.workload.weight_samples,
        cfg.workload.input_samples_per_weight,
        cfg.workload.batch_w,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, outer_margin=%d, inner_margin=%d",
        cfg.sweep.ratio_threshold,
        cfg.sweep.reltol,
        cfg.sweep.outer_margin,
        cfg.sweep.inner_margin,
    )
    log.info("=" * 80)

    # Build host xbar; solvers swap per candidate. The tool-run TOML's
    # ``[cim_macro]`` / ``[cim_macro_policy]`` sections use ``_neurox_use``
    # so any scheme's chip config + noise-off policy resolve transparently
    # through ``from_file``; the registry dispatches the concrete macro.
    stub = NestedParallelRailSolverConfig(n_outer=1, n_inner=1)
    xbar = build_xbar_for_calibration(
        cfg.cim_macro,
        cfg.cim_macro_policy,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
        solver_config=stub,
        solve_chunk_size=cfg.runtime.solve_chunk_size,
    )

    # --- Stage A: sweep n_outer at n_inner = inner_ref ---

    log.info("Stage A: sweep n_outer with n_inner pinned at %d", cfg.sweep.inner_ref)
    log.info("-" * 80)
    outer_candidates = _build_candidates(
        axis="n_outer",
        candidates=cfg.sweep.outer_candidates,
        other_value=cfg.sweep.inner_ref,
    )
    outer_rows, outer_scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=outer_candidates,
        n_weight=cfg.workload.weight_samples,
        n_input_per_weight=cfg.workload.input_samples_per_weight,
        batch_w=cfg.workload.batch_w,
        distribution_path=distribution_path,
        device=device,
        seed=cfg.runtime.seed,
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
        ratio_threshold=cfg.sweep.ratio_threshold,
        reltol=cfg.sweep.reltol,
    )
    if outer_pick.iter_count is None:
        log.error("Stage A failed: %s", outer_pick.reason)
        return 2
    pick_outer = outer_pick.iter_count
    log.info("Stage A pick: n_outer = %d  (%s)", pick_outer, outer_pick.reason)
    log.info("")
    if args.plot_dir is not None:
        plot_stage(
            outer_rows,
            outer_scale,
            axis_label="n_outer",
            out_path=args.plot_dir / "nested_stageA_outer.png",
            reltol=cfg.sweep.reltol,
        )

    # --- Stage B: sweep n_inner at n_outer = pick_outer ---

    log.info("=" * 80)
    log.info("Stage B: sweep n_inner with n_outer pinned at %d", pick_outer)
    log.info("-" * 80)
    inner_candidates = _build_candidates(
        axis="n_inner",
        candidates=cfg.sweep.inner_candidates,
        other_value=pick_outer,
    )
    inner_rows, inner_scale = aggregate_xbar_sweep(
        xbar,
        candidate_solvers=inner_candidates,
        n_weight=cfg.workload.weight_samples,
        n_input_per_weight=cfg.workload.input_samples_per_weight,
        batch_w=cfg.workload.batch_w,
        distribution_path=distribution_path,
        device=device,
        seed=cfg.runtime.seed,
    )
    log.info("")
    for r in inner_rows:
        log.info(_format_row(r, label="n_inner"))
    log.info("")
    inner_pick = pick_with_plateau_and_guard(
        inner_rows,
        inner_scale,
        ratio_threshold=cfg.sweep.ratio_threshold,
        reltol=cfg.sweep.reltol,
    )
    if inner_pick.iter_count is None:
        # n_inner can plausibly plateau at the smallest candidate — every iter
        # changes u so geometrically that the sweep's 2-point ratio test
        # cannot resolve the descent. Fall back to the smallest candidate but
        # re-verify the residual guard AT THAT candidate (Stage A's guard
        # was at n_inner_ref, which is generous; the smallest n_inner might
        # not meet residuals on its own).
        from neurox.tools._plateau import check_residual_relative_guard

        fallback = cfg.sweep.inner_candidates[0]
        fallback_row = next(r for r in inner_rows if r.iter_count == fallback)
        passed, ratios = check_residual_relative_guard(fallback_row, inner_scale, reltol=cfg.sweep.reltol)
        if not passed:
            worst = max(ratios.items(), key=lambda kv: kv[1])
            log.error(
                "Stage B plateau not detected AND fallback n_inner=%d fails residual guard: "
                "%s ratio=%.3e >= reltol=%.1e. Widen [sweep].inner_candidates or raise [sweep].reltol.",
                fallback,
                worst[0],
                worst[1],
                cfg.sweep.reltol,
            )
            return 2
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
            reltol=cfg.sweep.reltol,
        )

    final_outer = pick_outer + cfg.sweep.outer_margin
    final_inner = pick_inner + cfg.sweep.inner_margin
    log.info("=" * 80)
    log.info("Picked pair: (n_outer=%d, n_inner=%d)", pick_outer, pick_inner)
    log.info("Recommended with margins: (n_outer=%d, n_inner=%d)", final_outer, final_inner)
    log.info("Residual guard ratios at outer pick:")
    for k, v in outer_pick.residual_guard_ratios.items():
        log.info("    %-12s  %.3e  (reltol = %.1e)", k, v, cfg.sweep.reltol)
    log.info("")
    log.info("TOML fragment for chip preset:")
    log.info("    [cim_macro.array_config.solver_config]")
    log.info('    _neurox_class = "NestedParallelRailSolverConfig"')
    log.info("    n_outer = %d", final_outer)
    log.info("    n_inner = %d", final_inner)
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
