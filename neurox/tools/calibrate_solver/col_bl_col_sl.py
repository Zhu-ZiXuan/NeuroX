"""Calibrate the nested parallel-rail solver's `(n_outer, n_inner)` pair.

Two-axis sweep on primary step-ratio plateau detection plus a sanity relative
residual guard, staged:

  Stage A: fix `n_inner` at a generous reference, sweep `n_outer`
           -> pick the smallest `n_outer` at the step plateau.

  Stage B: fix `n_outer` at the Stage A pick, sweep `n_inner`
           -> pick the smallest `n_inner` at the step plateau.

Host-agnostic: the tool binds only to its calibration target — the nested
solver family and its cell-residual channel — and the abstract `CimMacro`
surface. Each candidate is a fresh macro rebuilt from the macro config file
with the swept iteration count patched onto the nested-solver table located by
`[macro].solver_section`; the workload rides the public `vec_mat_mul` over
serialized row planes and the calibration data is captured by the solver and
cell probers upstream of the ADC. Both criteria are chip-parameter-free.

CLI: `python -m neurox.tools.calibrate_solver.col_bl_col_sl --help`
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.common import ConfigBase
from neurox.primitive.macro.cim import CimMacroConfig, CimMacroPolicy
from neurox.tools._config import (
    add_standard_args,
    load_tool_config,
    resolve_relative_path,
    setup_logging,
)
from neurox.tools._plateau import CandidateRow, WorkloadScale, pick_with_plateau_and_guard

from ._common import (
    MacroSection,
    SolverSweepContext,
    aggregate_solver_sweep,
    build_calibration_macro,
    load_macro_config_dict,
    resolve_macro_files,
)


@dataclass(frozen=True)
class _WorkloadCfg:
    active_inputs: int
    """Simultaneously active logical inputs per serialized sub-phase plane,
    `1 <= active_inputs <= input_num`. The macro's `max_active_num` gives the
    production-faithful operating point and `input_num` the conservative
    single-plane envelope; any in-range value is legal, and the choice is never
    defaulted in code."""
    weight_samples: int
    """Distinct programmed weights to sweep."""
    input_samples_per_weight: int
    """Input vectors per weight, driven as one VMM batch."""
    batch_w: int
    """Weight-axis chunk size for the sampler."""
    distribution: Path | None = None
    """Synthetic-workload distribution TOML; absent means uniform sampling."""


@dataclass(frozen=True)
class _SweepCfg:
    outer_candidates: list[int]
    """Outer iteration counts swept in Stage A, ascending."""
    inner_candidates: list[int]
    """Inner iteration counts swept in Stage B, ascending; the first entry is
    also the Stage B fallback when no plateau is detected."""
    inner_ref: int
    """Generous inner count pinned throughout Stage A."""
    ratio_threshold: float
    """Step-ratio plateau threshold."""
    reltol: float
    """Relative residual-guard tolerance."""
    outer_margin: int
    """Added to the Stage A pick to form the recommended count."""
    inner_margin: int
    """Added to the Stage B pick to form the recommended count."""


@dataclass(frozen=True)
class _RuntimeCfg:
    dtype: str
    seed: int


class CalibrateSolverColBlColSlConfig(ConfigBase):
    macro: MacroSection
    workload: _WorkloadCfg
    sweep: _SweepCfg
    runtime: _RuntimeCfg


log = logging.getLogger(__name__)


def _format_row(row: CandidateRow, label: str) -> str:
    step = f"{row.step_max__V:9.2e}" if row.step_max__V is not None else "     ---"
    cell = row.residual_max.get("cell__uA")
    cell_str = f"{cell:9.2e}" if cell is not None else "      n/a"
    return (
        f"{label}={row.iter_count:3d}  "
        f"step={step}  "
        f"|F|.cell={cell_str}  "
        f"wire_bl={row.residual_max['f_bl_kcl__uA']:9.2e}  "
        f"clamp_bl={row.residual_max['f_bl_clamp__V']:9.2e}"
    )


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
    ax_step.set_ylabel("max |u_n - u_{n-1}| [V]")
    ax_step.set_title(f"Solution step ({axis_label} sweep)")
    ax_step.grid(True, which="both", ls=":", lw=0.4)

    for key, color in (("cell__uA", "C0"), ("f_bl_kcl__uA", "C1"), ("f_sl_kcl__uA", "C2")):
        ys = [r.residual_max.get(key, 0.0) for r in rows]
        ax_curr.plot(xs_all, ys, marker=".", color=color, label=key)
    ax_curr.axhline(
        reltol * scale.i_cell_typ__uA,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} x max|I_cell| = {reltol * scale.i_cell_typ__uA:.2e} uA)",
    )
    ax_curr.set_yscale("log")
    ax_curr.set_xlabel(axis_label)
    ax_curr.set_ylabel("max |residual| [uA]")
    ax_curr.set_title("Current residuals")
    ax_curr.grid(True, which="both", ls=":", lw=0.4)
    ax_curr.legend(fontsize="x-small", loc="upper right")

    for key, color in (("f_bl_clamp__V", "C3"), ("f_sl_clamp__V", "C4")):
        ys = [r.residual_max.get(key, 0.0) for r in rows]
        ax_volt.plot(xs_all, ys, marker=".", color=color, label=key)
    ax_volt.axhline(
        reltol * scale.v_node_typ__V,
        ls="--",
        color="gray",
        lw=0.7,
        label=f"guard ({reltol:.1e} x max|V_node| = {reltol * scale.v_node_typ__V:.2e} V)",
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
        description="Calibrate the parallel BL/SL solver (n_outer, n_inner) via step-ratio plateau."
    )
    add_standard_args(parser, plot_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(CalibrateSolverColBlColSlConfig, args.config)
    log.info("loaded config from %s", args.config)

    inst_shape = (cfg.workload.batch_w,)
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    # Resolve the macro files once; the base dict is patched per candidate and
    # the sampling host is a single macro built from the shipped config.
    config_paths, policy_path = resolve_macro_files(cfg.macro, base=args.config)
    base_macro_dict = load_macro_config_dict(config_paths, config_section=cfg.macro.config_section)
    policy = CimMacroPolicy.from_file(policy_path, section=cfg.macro.policy_section)
    base_config = CimMacroConfig.from_dict(base_macro_dict)
    sampling_host = build_calibration_macro(
        base_config,
        policy,
        device=device,
        inst_shape=inst_shape,
        dtype=dtype,
    )

    active_inputs = cfg.workload.active_inputs
    if not (1 <= active_inputs <= sampling_host.input_num):
        raise SystemExit(
            f"[workload].active_inputs ({active_inputs}) must satisfy "
            f"1 <= active_inputs <= input_num ({sampling_host.input_num})."
        )

    distribution = cfg.workload.distribution
    distribution_path = None if distribution is None else resolve_relative_path(distribution, args.config)

    log.info("=" * 80)
    log.info("Parallel BL/SL solver — step-ratio plateau calibration (2-axis staged)")
    log.info(
        "workload: inst=%s, %d weights x %d inputs (batch_w=%d), active_inputs=%d of input_num=%d",
        inst_shape,
        cfg.workload.weight_samples,
        cfg.workload.input_samples_per_weight,
        cfg.workload.batch_w,
        active_inputs,
        sampling_host.input_num,
    )
    log.info(
        "criteria: ratio_threshold=%.3f, reltol=%.1e, outer_margin=%d, inner_margin=%d",
        cfg.sweep.ratio_threshold,
        cfg.sweep.reltol,
        cfg.sweep.outer_margin,
        cfg.sweep.inner_margin,
    )
    log.info("=" * 80)

    sweep_context = SolverSweepContext(
        base_macro_dict=base_macro_dict,
        solver_section=cfg.macro.solver_section,
        policy=policy,
        sampling_host=sampling_host,
        inst_shape=inst_shape,
        dtype=dtype,
        active_inputs=active_inputs,
        n_weight=cfg.workload.weight_samples,
        n_input_per_weight=cfg.workload.input_samples_per_weight,
        batch_w=cfg.workload.batch_w,
        distribution_path=distribution_path,
        device=device,
        seed=cfg.runtime.seed,
    )

    # --- Stage A: sweep n_outer at n_inner = inner_ref ---

    log.info("Stage A: sweep n_outer with n_inner pinned at %d", cfg.sweep.inner_ref)
    log.info("-" * 80)
    outer_result = aggregate_solver_sweep(
        swept_key="n_outer",
        candidates=cfg.sweep.outer_candidates,
        fixed_overrides={"n_inner": cfg.sweep.inner_ref},
        context=sweep_context,
    )
    outer_rows = outer_result.rows
    outer_scale = outer_result.scale
    log.info(
        "workload scale: max|I_cell|=%.3e uA, max|V_BL_node|=%.3e V",
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
    inner_result = aggregate_solver_sweep(
        swept_key="n_inner",
        candidates=cfg.sweep.inner_candidates,
        fixed_overrides={"n_outer": pick_outer},
        context=sweep_context,
    )
    inner_rows = inner_result.rows
    inner_scale = inner_result.scale
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
        # n_inner can plateau at the smallest candidate, the descent too steep
        # for the sweep's 2-point ratio test to resolve. Fall back to it, but
        # re-verify the residual guard THERE: Stage A verified it at the
        # generous inner_ref, which the smallest n_inner need not match.
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
    log.info("TOML fragment for the solver table at %s:", cfg.macro.solver_section)
    log.info("    n_outer = %d", final_outer)
    log.info("    n_inner = %d", final_inner)
    log.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
