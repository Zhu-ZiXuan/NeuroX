"""Forward TIA design tool — grid-search candidates against a workload.

CLI: ``python -m neurox.tools.xbar_tia.optimize --config <design.toml>
[--plot <out.png>] [--log-level INFO]``

All inputs come from the TOML config (no chip preset, no input-carrying CLI flags):

  - ``[hardware]`` — externally fixed: ``v_dd__V``, ``v_ref__V``,
    ``output_saturation_softness__V``, ``target_v_max__V``, ``tia_n_newton``,
    plus an embedded ``[hardware.nmos_config]`` table that may use the
    ``_neurox_use_preset`` pattern to point at a process file.
  - ``[workload]`` — the Gaussian current model: ``mean__uA``, ``std__uA``.
  - ``[sweep]`` — lists for each design knob: ``opamp_gain``,
    ``pseudo_nmos_W__um``, ``pseudo_nmos_L__um``, ``v_nmos_bias__V``.
    Cartesian product over all four axes.

Scoring (higher = better; 0 for infeasible candidates):
  ``score = linearity_r2 × range_use × overshoot_safe``, all measured
  against the user's ``target_v_max__V``: ``linearity_r2`` is the Pearson
  R² of ``v_out`` vs ``I`` over the workload ±3σ band; ``range_use`` =
  ``min(1, (v(μ+3σ) - v(μ-3σ)) / target_v_max__V)`` is the fraction of the
  target output span that band fills; ``overshoot_safe`` falls linearly
  from 1 to 0 as ``v(μ+3σ)`` climbs from ``target_v_max__V`` toward the
  chip rail — output past the ceiling is penalised.

Output: top-K candidates printed to log with the raw (gain, W, L, v_bias)
tuple; an optional top-K curve overlay PNG (``--plot``) and optional
per-axis slice PNGs (``--slice-plot-dir``).
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.analog.tia import OpAmpTIAConfig
from neurox.device import MOSFETConfig
from neurox.tools._config import add_standard_args, load_tool_config, setup_logging
from neurox.tools.xbar_tia._common import (
    TransferCurve,
    build_tia,
    fit_to_workload,
    linearity_r2,
    sweep_transfer,
)

logger = logging.getLogger(__name__)


# --- config schema ----------------------------------------------------------


@dataclass(frozen=True)
class HardwareSection:
    """Chip-level constants that the design tool does NOT optimise over.

    Attributes:
        nmos_config: Pseudo-resistor NMOS process config.
        v_dd__V: Chip supply rail — physical hard cap for TIA output.
        v_ref__V: Reference voltage of the BL clamp.
        output_saturation_softness__V: Softclip softness band.
        target_v_max__V: User-chosen ceiling for the TIA output that
            aligns with the downstream ADC's largest v_ref mode
            (chip's softclip upper rail = ``v_dd`` is the physical hard
            cap, but the user typically wants ``v_out`` to stay below
            the ADC's biggest v_ref to maximise usable signal span
            without rail clipping).
        tia_n_newton: Newton iteration count for the inner TIA solve.
    """

    nmos_config: MOSFETConfig
    v_dd__V: float
    v_ref__V: float
    output_saturation_softness__V: float
    target_v_max__V: float
    tia_n_newton: int


@dataclass(frozen=True)
class WorkloadSection:
    """Gaussian model of the per-column workload current.

    Attributes:
        mean__uA: Mean BL port current of the modelled workload.
            Must be ≥ 0 — the TIA sweeps a non-negative input grid.
        std__uA: Standard deviation; must be ≥ 0.
    """

    mean__uA: float
    std__uA: float

    def __post_init__(self) -> None:
        if self.mean__uA < 0.0:
            raise ValueError(f"[workload].mean__uA ({self.mean__uA}) must be >= 0")
        if self.std__uA < 0.0:
            raise ValueError(f"[workload].std__uA ({self.std__uA}) must be >= 0")


@dataclass(frozen=True)
class SweepSection:
    """Per-axis value lists; cartesian product enumerates candidates."""

    opamp_gain: tuple[float, ...]
    pseudo_nmos_W__um: tuple[float, ...]
    pseudo_nmos_L__um: tuple[float, ...]
    v_nmos_bias__V: tuple[float, ...]


@dataclass(frozen=True)
class TiaDesignConfig:
    """Top-level config consumed by ``optimize.py``."""

    hardware: HardwareSection
    workload: WorkloadSection
    sweep: SweepSection

    def __post_init__(self) -> None:
        if not (self.hardware.target_v_max__V > 0.0):
            raise ValueError(f"[hardware].target_v_max__V ({self.hardware.target_v_max__V}) must be > 0")


# --- evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    opamp_gain: float
    pseudo_nmos_W__um: float
    pseudo_nmos_L__um: float
    v_nmos_bias__V: float
    curve: TransferCurve
    v_at_mean__V: float
    v_at_lo3sigma__V: float
    v_at_hi3sigma__V: float
    slope_at_mean__mV_per_uA: float
    saturation_onset__uA: float
    sat_margin_above_3sigma__uA: float
    linearity_r2: float
    range_use: float
    overshoot_safe: float
    score: float
    is_feasible: bool  # False if slope(μ)~0 or μ pinned at high rail


def _build_tia_config(hw: HardwareSection, gain: float, w: float, nmos_L_um: float, vb: float) -> OpAmpTIAConfig:
    """Stitch a per-combo :class:`OpAmpTIAConfig`.

    The reference clamp voltage is injected per call into the TIA's snap
    (see :func:`sweep_transfer`); ``hw.v_ref__V`` is consumed there.
    """
    return OpAmpTIAConfig(
        v_nmos_bias__V=vb,
        v_dd__V=hw.v_dd__V,
        opamp_gain=gain,
        opamp_gain_sigma=0.0,
        pseudo_nmos_W__um=w,
        pseudo_nmos_L__um=nmos_L_um,
        output_saturation_softness__V=hw.output_saturation_softness__V,
        n_newton=hw.tia_n_newton,
        nmos_config=hw.nmos_config,
        # PPA fields are irrelevant to the DC-transfer design analysis.
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _evaluate(
    hw: HardwareSection,
    workload: WorkloadSection,
    gain: float,
    w: float,
    nmos_L_um: float,
    vb: float,
    *,
    i_max_uA: float,
    n_points: int,
    device: torch.device,
) -> CandidateResult | None:
    """Always-returns evaluation — keeps the curve even for infeasible combos
    so 1D-slice plots can show the full grid. Returns ``None`` only when the
    combo violates the pseudo-resistor design constraint (Vb ≤ v_ref).

    Scoring targets ``hw.target_v_max__V``. Goal: workload ±3σ span maps linearly
    into a v_out range that fills ``[0, target_v_max]`` without spilling
    past it.
    """
    if vb <= hw.v_ref__V:
        return None  # pseudo-resistor needs gate overdrive above the source clamp; skip gracefully
    cfg = _build_tia_config(hw, gain, w, nmos_L_um, vb)
    tia = build_tia(cfg, device=device)
    curve = sweep_transfer(tia, v_ref__V=hw.v_ref__V, i_min_uA=0.0, i_max_uA=i_max_uA, n_points=n_points, device=device)
    fit = fit_to_workload(curve, mean_uA=workload.mean__uA, std_uA=workload.std__uA)
    target_v = hw.target_v_max__V
    i_lo = max(0.0, workload.mean__uA - 3 * workload.std__uA)
    i_hi = workload.mean__uA + 3 * workload.std__uA
    v_lo = curve.v_at(i_lo)
    v_hi = curve.v_at(i_hi)
    # Linearity *in the workload band only* — not the whole pre-saturation curve.
    r2_workload = linearity_r2(curve, lo_uA=i_lo, hi_uA=i_hi)
    # range_use: how much of [0, target_v_max] the workload band occupies.
    workload_v_span = v_hi - v_lo
    range_use = min(1.0, workload_v_span / target_v) if workload_v_span > 0 else 0.0
    # overshoot_safe: linear penalty as v_hi climbs from target_v_max toward chip rail.
    rail_headroom = curve.v_max_V - target_v
    overshoot = max(0.0, v_hi - target_v)
    overshoot_safe = max(0.0, 1.0 - overshoot / rail_headroom) if rail_headroom > 0 else 0.0
    score = r2_workload * range_use * overshoot_safe
    is_feasible = (
        fit.slope_at_mean__mV_per_uA > 1e-3
        and fit.v_at_mean__V < curve.v_max_V - 0.02
        and v_hi < curve.v_max_V - 0.001  # workload +3σ tail not pinned at hard rail
    )
    return CandidateResult(
        opamp_gain=gain,
        pseudo_nmos_W__um=w,
        pseudo_nmos_L__um=nmos_L_um,
        v_nmos_bias__V=vb,
        curve=curve,
        v_at_mean__V=fit.v_at_mean__V,
        v_at_lo3sigma__V=v_lo,
        v_at_hi3sigma__V=v_hi,
        slope_at_mean__mV_per_uA=fit.slope_at_mean__mV_per_uA,
        saturation_onset__uA=fit.saturation_onset__uA,
        sat_margin_above_3sigma__uA=fit.sat_margin_above_3sigma__uA,
        linearity_r2=r2_workload,
        range_use=range_use,
        overshoot_safe=overshoot_safe,
        score=score if is_feasible else 0.0,
        is_feasible=is_feasible,
    )


# --- output -----------------------------------------------------------------


def _log_candidate(prefix: str, r: CandidateResult) -> None:
    logger.info(
        "%s A=%g W=%g L=%g Vb=%g  | R^2_wl=%.3f  range_use=%.2f  overshoot_safe=%.2f"
        "  | v(-3sigma)=%.3fV  v(mu)=%.3fV  v(+3sigma)=%.3fV  slope(mu)=%.2f mV/uA  score=%.4f",
        prefix,
        r.opamp_gain,
        r.pseudo_nmos_W__um,
        r.pseudo_nmos_L__um,
        r.v_nmos_bias__V,
        r.linearity_r2,
        r.range_use,
        r.overshoot_safe,
        r.v_at_lo3sigma__V,
        r.v_at_mean__V,
        r.v_at_hi3sigma__V,
        r.slope_at_mean__mV_per_uA,
        r.score,
    )


def _plot_slice(
    slice_candidates: list[CandidateResult],
    *,
    varied_attr: str,
    fixed_attrs_label: str,
    workload: WorkloadSection,
    output_path: Path,
) -> None:
    """One PNG: every candidate on the slice plotted as v_out(I); legend shows
    the varied attribute and per-curve metrics. Infeasible curves are drawn
    dashed and dimmer so they stay visible without polluting the readability."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for slice plots but is not installed") from exc
    short = {
        "opamp_gain": "A",
        "pseudo_nmos_W__um": "W",
        "pseudo_nmos_L__um": "L",
        "v_nmos_bias__V": "Vb",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    slice_candidates = sorted(slice_candidates, key=lambda r: getattr(r, varied_attr))
    for idx, r in enumerate(slice_candidates):
        color = cmap(idx / max(len(slice_candidates) - 1, 1))
        ls = "-" if r.is_feasible else "--"
        alpha = 1.0 if r.is_feasible else 0.45
        feasibility_tag = "" if r.is_feasible else " [infeasible]"
        # Unit-glyph LOCAL EXCEPTION to the global units-ASCII convention (a
        # local exception overrides the global rule): matplotlib plot strings
        # spell units as unicode glyphs per scientific-figure convention — µ
        # (U+00B5) in "µA" here (Ω, °, µm² elsewhere) — while math variables
        # stay LaTeX. Two mu's: mean = LaTeX $\mu$, micro-prefix "µA" = glyph.
        label = (
            f"{short.get(varied_attr, varied_attr)}={getattr(r, varied_attr):g}{feasibility_tag}  "
            rf"$R^2$={r.linearity_r2:.3f}  slope={r.slope_at_mean__mV_per_uA:.2f}mV/µA  "
            rf"v($\mu$)={r.v_at_mean__V:.3f}V  score={r.score:.3f}"
        )
        ax.plot(
            r.curve.i_uA.numpy(),
            r.curve.v_out_V.numpy(),
            color=color,
            linewidth=1.4,
            linestyle=ls,
            alpha=alpha,
            label=label,
        )
    rails = slice_candidates[0].curve
    ax.axhline(rails.v_min_V, color="grey", linestyle=":", linewidth=0.7)
    ax.axhline(rails.v_max_V, color="grey", linestyle=":", linewidth=0.7)
    ax.axvspan(
        workload.mean__uA - 3 * workload.std__uA,
        workload.mean__uA + 3 * workload.std__uA,
        color="tab:blue",
        alpha=0.10,
        label=r"workload $\mu \pm 3\sigma$",
    )
    ax.axvline(workload.mean__uA, color="tab:blue", linestyle="--", linewidth=0.8)
    # Axis unit as unicode glyph "µA": units-ASCII LOCAL EXCEPTION (the $I$ variable stays LaTeX).
    ax.set_xlabel(r"$I_{\mathrm{port}}$ [µA]")
    ax.set_ylabel(r"$v_{\mathrm{out}}$ [V]")
    ax.set_title(f"TIA slice — vary {short.get(varied_attr, varied_attr)} (fixed: {fixed_attrs_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _emit_slice_plots(
    results: list[CandidateResult],
    workload: WorkloadSection,
    output_dir: Path,
) -> None:
    """Per (gain, L, Vb): a W-axis sweep PNG. Per (gain, L, W): a Vb-axis sweep PNG.
    `gain` and `L` are treated as fixed-only axes (never the varied axis), per the
    user's design intent — they're chip-design choices, not workload-tuning knobs."""
    by_key = {(r.opamp_gain, r.pseudo_nmos_W__um, r.pseudo_nmos_L__um, r.v_nmos_bias__V): r for r in results}
    gains = sorted({r.opamp_gain for r in results})
    Ws = sorted({r.pseudo_nmos_W__um for r in results})
    Ls = sorted({r.pseudo_nmos_L__um for r in results})
    Vbs = sorted({r.v_nmos_bias__V for r in results})

    w_dir = output_dir / "slice_W"
    vb_dir = output_dir / "slice_Vb"
    w_dir.mkdir(parents=True, exist_ok=True)
    vb_dir.mkdir(parents=True, exist_ok=True)

    n_w_plots = 0
    n_vb_plots = 0

    for g in gains:
        for nmos_L_um in Ls:
            for vb in Vbs:
                raw_slice = [by_key.get((g, w, nmos_L_um, vb)) for w in Ws]
                filtered_slice: list[CandidateResult] = [r for r in raw_slice if r is not None]
                if len(filtered_slice) < 2:
                    # Skip degenerate slices (one or zero valid candidates
                    # left after invalid combos were dropped) — there is
                    # nothing for a sweep plot to show.
                    continue
                fn = w_dir / f"sweep_W__gain{g:g}_L{nmos_L_um:g}_Vb{vb:g}.png"
                _plot_slice(
                    filtered_slice,
                    varied_attr="pseudo_nmos_W__um",
                    fixed_attrs_label=f"A={g:g}, L={nmos_L_um:g}, Vb={vb:g}",
                    workload=workload,
                    output_path=fn,
                )
                n_w_plots += 1
            for w in Ws:
                raw_slice = [by_key.get((g, w, nmos_L_um, vb)) for vb in Vbs]
                filtered_slice = [r for r in raw_slice if r is not None]
                if len(filtered_slice) < 2:
                    continue
                fn = vb_dir / f"sweep_Vb__gain{g:g}_L{nmos_L_um:g}_W{w:g}.png"
                _plot_slice(
                    filtered_slice,
                    varied_attr="v_nmos_bias__V",
                    fixed_attrs_label=f"A={g:g}, L={nmos_L_um:g}, W={w:g}",
                    workload=workload,
                    output_path=fn,
                )
                n_vb_plots += 1

    logger.info("wrote %d W-axis slice plots to %s/", n_w_plots, w_dir)
    logger.info("wrote %d Vb-axis slice plots to %s/", n_vb_plots, vb_dir)


def _plot_top_k(top: list[CandidateResult], workload: WorkloadSection, output_path: Path) -> None:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot but is not installed") from exc
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    for idx, r in enumerate(top):
        color = cmap(idx / max(len(top) - 1, 1))
        lw = 1.8 if idx == 0 else 1.2
        # Unit-glyph LOCAL EXCEPTION to the global units-ASCII convention (a
        # local exception overrides the global rule): matplotlib plot strings
        # spell units as unicode glyphs per scientific-figure convention — µ
        # (U+00B5) in "µA" here (Ω, °, µm² elsewhere) — while the math var
        # $R^2$ stays LaTeX.
        label = (
            f"#{idx + 1} A={r.opamp_gain:g} W={r.pseudo_nmos_W__um:g} "
            f"L={r.pseudo_nmos_L__um:g} Vb={r.v_nmos_bias__V:g}  "
            rf"$R^2$={r.linearity_r2:.3f}  slope={r.slope_at_mean__mV_per_uA:.2f}mV/µA  score={r.score:.3f}"
        )
        ax.plot(r.curve.i_uA.numpy(), r.curve.v_out_V.numpy(), color=color, linewidth=lw, label=label)
    ax.axhline(top[0].curve.v_min_V, color="grey", linestyle=":", linewidth=0.7)
    ax.axhline(top[0].curve.v_max_V, color="grey", linestyle=":", linewidth=0.7)
    ax.axvspan(
        workload.mean__uA - 3 * workload.std__uA,
        workload.mean__uA + 3 * workload.std__uA,
        color="tab:blue",
        alpha=0.10,
        label=r"workload $\mu \pm 3\sigma$",
    )
    ax.axvline(workload.mean__uA, color="tab:blue", linestyle="--", linewidth=0.8)
    # Axis unit as unicode glyph "µA": units-ASCII LOCAL EXCEPTION (the $I$ variable stays LaTeX).
    ax.set_xlabel(r"$I_{\mathrm{port}}$ [µA]")
    ax.set_ylabel(r"$v_{\mathrm{out}}$ [V]")
    ax.set_title("Top-K TIA candidates — score = linearity_r2 · range_use · overshoot_safe")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote plot to %s", output_path)


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward TIA design tool (config-driven)")
    add_standard_args(parser, plot_file=True)
    parser.add_argument(
        "--slice-plot-dir",
        type=Path,
        default=None,
        help="Optional output directory for 1D slice PNGs (one per fixed (gain, L, Vb) and "
        "(gain, L, W) combo, varying W and Vb respectively)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top candidates to report (default 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    device = torch.device(args.device)

    cfg = load_tool_config(TiaDesignConfig, args.config)
    logger.info("loaded TIA design config from %s", args.config)
    logger.info(
        "  hardware: v_dd=%.3fV  v_ref=%.3fV  softness=%.3fV  (nmos via preset)",
        cfg.hardware.v_dd__V,
        cfg.hardware.v_ref__V,
        cfg.hardware.output_saturation_softness__V,
    )
    logger.info("  workload: N(mean=%.1f, std=%.1f) uA", cfg.workload.mean__uA, cfg.workload.std__uA)

    axes = {
        "opamp_gain": cfg.sweep.opamp_gain,
        "pseudo_nmos_W__um": cfg.sweep.pseudo_nmos_W__um,
        "pseudo_nmos_L__um": cfg.sweep.pseudo_nmos_L__um,
        "v_nmos_bias__V": cfg.sweep.v_nmos_bias__V,
    }
    for name, vs in axes.items():
        if not vs:
            raise SystemExit(f"[sweep].{name} is empty; every axis must have at least one value")
    n_total = 1
    for vs in axes.values():
        n_total *= len(vs)
    logger.info("  sweep grid: %s -> %d combos", {k: len(v) for k, v in axes.items()}, n_total)

    i_max = max(1500.0, 1.6 * (cfg.workload.mean__uA + 5 * cfg.workload.std__uA))

    all_results: list[CandidateResult] = []
    skipped_invalid = 0
    for gain, w, nmos_L_um, vb in itertools.product(
        cfg.sweep.opamp_gain,
        cfg.sweep.pseudo_nmos_W__um,
        cfg.sweep.pseudo_nmos_L__um,
        cfg.sweep.v_nmos_bias__V,
    ):
        r = _evaluate(
            cfg.hardware,
            cfg.workload,
            gain,
            w,
            nmos_L_um,
            vb,
            i_max_uA=i_max,
            n_points=301,
            device=device,
        )
        if r is None:
            skipped_invalid += 1
            continue
        all_results.append(r)
    feasible = [r for r in all_results if r.is_feasible]
    logger.info(
        "scanned %d combos; %d valid; %d feasible, %d infeasible; %d skipped (Vb≤v_ref)",
        n_total,
        len(all_results),
        len(feasible),
        len(all_results) - len(feasible),
        skipped_invalid,
    )
    if not feasible:
        raise SystemExit("no feasible candidate in the grid — widen [sweep] ranges")

    feasible.sort(key=lambda r: r.score, reverse=True)
    top = feasible[: args.top_k]
    logger.info("")
    logger.info("top %d candidates by score:", len(top))
    for rank, r in enumerate(top, start=1):
        _log_candidate(f"  #{rank:>2}", r)

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _plot_top_k(top, cfg.workload, args.plot)
    if args.slice_plot_dir is not None:
        args.slice_plot_dir.mkdir(parents=True, exist_ok=True)
        _emit_slice_plots(all_results, cfg.workload, args.slice_plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
