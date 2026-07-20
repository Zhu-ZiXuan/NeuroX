"""CLI: probe the analog I(M) grid of a CIM macro and place ADC thresholds.

CLI: ``python -m neurox.tools.calibrate_adc.threshold_probe --config <run.toml>
[--device cuda:N] [--output <fragment.toml>] [--plot-dir <dir>]
[--log-dir <dir>] [--log-level INFO] [--element-range a:b]
[--capture-out <part.pt>] [--capture-in <part.pt>[,<part.pt>...]]``

Controlled-stimulus grid sweep: a deterministic count-grid battery (every
per-(column, phase)-block magnitude ``0 .. m_max`` realized by
single-cell-LSB patterns under full WL drive), count-capped random
single-sign block patterns, dense saturating columns, and random WL drive
densities. Each battery element runs through the physical tile and its
lossless twin under an :class:`~neurox.common.prober.AdcProber`; the
``adc.convert`` captured analog inputs pair with the ``adc.ideal_vmm``
integer dots, giving the observed analog band per ``|M|``. The modes come
from the mode-set TOML named by the run config (``modes_file``, see
:mod:`._modes`); per mode the ladder top is ``m_max = ceil(range)`` (the
ladder must cover the mode's design range). Per mode the tool places the
mid-point threshold ladder
``t[k] = (hi(k) + lo(k+1)) / 2``, reports the band margins (headline: the
minimum), and emits a ``ref_levels__uA`` / ``i_refs__uA`` row fragment
plus figures (grid curve with bands + thresholds, per-mode margin bars).

Capture staging bounds single-command runtime on large batteries: the
battery element list is deterministic for a given config, so
``--element-range a:b`` probes a contiguous slice and ``--capture-out``
saves that slice's pooled ``(|M|, i_in__uA)`` streams (skipping
placement). A later run merges every ``--capture-in`` part ahead of its
own probed slice and places the ladder over the union; the log records
the merged provenance.

See also:
    docs/guides/calibration/calibrate_adc.md
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.common import ConfigBase
from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.macro.cim.ideal import IdealCimMacro
from neurox.tools._config import add_standard_args, load_tool_config, resolve_relative_path, setup_logging
from neurox.tools.calibrate_adc._math import (
    MagnitudeBand,
    ThresholdPlacement,
    band_stats,
    fit_linear,
    place_thresholds,
)
from neurox.tools.calibrate_adc._modes import load_mode_set
from neurox.tools.calibrate_adc._testbench import (
    MacroSection,
    add_file_logging,
    build_ideal_twin,
    build_physical_macro,
    grid_block_w,
    run_paired_stimulus,
    sample_binary_x,
    sample_capped_block_w,
    saturating_w,
)

logger = logging.getLogger(__name__)


# --- config schema ----------------------------------------------------------


@dataclass(frozen=True)
class _StimulusCfg:
    """``[stimulus]`` section: the probing battery.

    Attributes:
        seed: RNG seed for the random battery elements.
        lsb_caps: Per-(column, phase)-block magnitude caps of the random
            single-sign count-capped patterns.
        patterns_per_cap: Independent random patterns per cap.
        x_densities: Random binary WL drive densities crossed with every
            random pattern (the deterministic grid / saturating patterns
            always run under full drive so their block counts are exact).
        full_drive_caps: Subset of ``lsb_caps`` whose random patterns ALSO
            run one full-drive element (exact block counts). On a wide
            tile a high-cap all-column pattern under full drive can exceed
            the solver-convergent loading envelope — list only the caps
            that stay inside it (exact high-count coverage then comes from
            the diluted grid).
        x_batch: Drive vectors per (pattern, density) combo.
        grid_col_stride: Column stride of the deterministic count grid —
            every ``grid_col_stride``-th column is programmed, the others
            stay zero. A loading-dilution knob: the exact-coverage grid
            runs under full WL drive, so on a wide tile a stride > 1 keeps
            the total array conduction inside the workload envelope the DC
            solve converges on (the per-column count stays exact).
        include_saturating: Add the dense +1 / -1 / phase-antisymmetric
            saturating-column pattern to the battery. Only meaningful when
            the tile's DC solve converges on that full-drive fully-dense
            extreme; outside that envelope the samples are invalid and the
            pattern must stay off.
    """

    seed: int
    lsb_caps: tuple[int, ...]
    patterns_per_cap: int
    x_densities: tuple[float, ...]
    full_drive_caps: tuple[int, ...]
    x_batch: int
    grid_col_stride: int
    include_saturating: bool

    def __post_init__(self) -> None:
        if not self.lsb_caps:
            raise ValueError("require: [stimulus].lsb_caps non-empty")
        if not set(self.full_drive_caps) <= set(self.lsb_caps):
            raise ValueError(
                f"require: full_drive_caps ({self.full_drive_caps}) is a subset of lsb_caps ({self.lsb_caps})"
            )
        if self.patterns_per_cap < 1:
            raise ValueError(f"require: [stimulus].patterns_per_cap ({self.patterns_per_cap}) >= 1")
        if self.x_batch < 1:
            raise ValueError(f"require: [stimulus].x_batch ({self.x_batch}) >= 1")
        if self.grid_col_stride < 1:
            raise ValueError(f"require: [stimulus].grid_col_stride ({self.grid_col_stride}) >= 1")
        for d in self.x_densities:
            if not (0.0 <= d <= 1.0):
                raise ValueError(f"require: x_densities in [0, 1]; got {d}")


@dataclass(frozen=True)
class _ProbeCfg:
    """``[probe]`` section: capture operating point.

    Attributes:
        adc_mode: Mode passed to the physical run while capturing. The
            captured analog input is mode-independent (mode selection is
            quasi-static reference switching downstream of the probe
            point), so one capture serves every ``[[modes]]`` placement.
    """

    adc_mode: int


@dataclass(frozen=True)
class ThresholdProbeToolConfig(ConfigBase):
    """Top-level config for :mod:`neurox.tools.calibrate_adc.threshold_probe`.

    Attributes:
        macro: The tile to build.
        probe: Capture operating point.
        stimulus: The probing battery.
        modes_file: Mode-set TOML (see
            :func:`~neurox.tools.calibrate_adc._modes.load_mode_set`),
            relative to the tool TOML; one ladder is placed per mode with
            ``m_max = ceil(range)``.
    """

    macro: MacroSection
    probe: _ProbeCfg
    stimulus: _StimulusCfg
    modes_file: Path


# --- probing ----------------------------------------------------------------


def _build_battery(
    physical: CimMacro,
    *,
    cfg: ThresholdProbeToolConfig,
    grid_top: int,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Materialize the deterministic battery element list ``(name, w, x)``.

    The list is a pure function of the config, the mode set's largest
    ``m_max`` (``grid_top``), and the tile geometry (the random elements
    draw from a seeded generator in enumeration order), so every
    ``--element-range`` slice of the same inputs sees the same elements.
    """
    stim = cfg.stimulus
    gen = torch.Generator().manual_seed(stim.seed)
    row_num = physical.row_num
    col_num = physical.col_num
    active_row_num = physical.max_active_rows

    full_drive = torch.ones((1, row_num), dtype=torch.long)

    # 1. Deterministic count grid (exact |M| coverage 0 .. grid_top, full
    # drive, columns diluted by grid_col_stride); a narrow tile walks the
    # grid over several offset patterns.
    grid_cols = max(col_num // stim.grid_col_stride, 1)
    grid_pattern_num = -(-(grid_top + 1) // grid_cols)
    batteries: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (
            f"grid_off{i * grid_cols}",
            grid_block_w(
                col_num=col_num,
                row_num=row_num,
                active_row_num=active_row_num,
                m_max=grid_top,
                offset=i * grid_cols,
                col_stride=stim.grid_col_stride,
            ),
            full_drive,
        )
        for i in range(grid_pattern_num)
    ]
    # 2. Count-capped random single-sign block patterns x drive densities
    # (full drive only for the caps declared inside the convergent envelope).
    for cap in stim.lsb_caps:
        for _ in range(stim.patterns_per_cap):
            w = sample_capped_block_w(gen, col_num=col_num, row_num=row_num, active_row_num=active_row_num, cap=cap)
            if cap in stim.full_drive_caps:
                batteries.append((f"cap{cap}_full", w, full_drive))
            for density in stim.x_densities:
                x = sample_binary_x(gen, batch=stim.x_batch, row_num=row_num, density=density)
                batteries.append((f"cap{cap}_d{density:g}", w, x))
    # 3. Dense saturating columns (loading-envelope extreme).
    if stim.include_saturating:
        batteries.append(
            ("saturating", saturating_w(col_num=col_num, row_num=row_num, active_row_num=active_row_num), full_drive)
        )
    return batteries


def _probe_grid(
    physical: CimMacro,
    ideal: IdealCimMacro,
    *,
    cfg: ThresholdProbeToolConfig,
    grid_top: int,
    element_range: tuple[int, int | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the battery slice; return pooled ``(|M|, i_in__uA)`` CPU streams."""
    op = AdcOperationPoint(adc_mode=cfg.probe.adc_mode, adc_bits=physical.adc_max_bits)
    batteries = _build_battery(physical, cfg=cfg, grid_top=grid_top)
    start, stop = element_range
    battery_slice = batteries[start:stop]
    logger.info(
        "battery elements %d..%d of %d",
        start,
        start + len(battery_slice) - 1,
        len(batteries),
    )
    m_parts: list[torch.Tensor] = []
    i_parts: list[torch.Tensor] = []
    for name, w, x in battery_slice:
        pair = run_paired_stimulus(physical, ideal, w=w, x=x, adc_operation_point=op)
        m_parts.append(pair.ideal_m.abs())
        i_parts.append(pair.i_in__uA)
        logger.info("battery %-14s -> %d conversion samples", name, pair.i_in__uA.numel())
    if not m_parts:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float64)
    return torch.cat(m_parts), torch.cat(i_parts)


# --- output -----------------------------------------------------------------


@dataclass(frozen=True)
class ModePlacement:
    """Placement + diagnostics for one ``[[modes]]`` entry."""

    adc_mode: int
    m_max: int
    bands: tuple[MagnitudeBand, ...]
    placement: ThresholdPlacement


def _fragment_lines(placements: list[ModePlacement]) -> list[str]:
    """The threshold-ladder fragment (mode rows, ascending ``adc_mode``)."""
    rows = [
        "[" + ", ".join(f"{t:.6f}" for t in p.placement.thresholds) + "]"
        for p in sorted(placements, key=lambda p: p.adc_mode)
    ]
    body = ",\n    ".join(rows)
    return [
        "# threshold ladders probed by neurox.tools.calibrate_adc.threshold_probe;",
        "# paste the same 2-D bank into BOTH adc_config.ref_levels__uA and",
        "# reference_config.i_refs__uA (row index = adc_mode).",
        "ref_levels__uA = [\n    " + body + ",\n]",
        "i_refs__uA = [\n    " + body + ",\n]",
    ]


def _log_mode(p: ModePlacement) -> None:
    logger.info("mode %d (m_max %d):", p.adc_mode, p.m_max)
    for band in p.bands:
        logger.info(
            "  |M| = %2d: %f .. %f  (mean %f, n = %d)",
            band.magnitude,
            band.lo,
            band.hi,
            band.mean,
            band.count,
        )
    logger.info("  thresholds__uA = [%s]", ", ".join(f"{t:.6f}" for t in p.placement.thresholds))
    logger.info("  margins__uA    = [%s]", ", ".join(f"{m:.6f}" for m in p.placement.margins))
    logger.info(
        "  min margin = %+f uA at the %d/%d boundary; monotone = %s",
        p.placement.min_margin,
        p.placement.min_margin_boundary,
        p.placement.min_margin_boundary + 1,
        p.placement.monotone,
    )


def _plot_grid_curve(p: ModePlacement, output_path: Path) -> None:
    """One PNG per mode: band envelope + means vs |M| with the placed thresholds."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mags = [b.magnitude for b in p.bands]
    los = [b.lo for b in p.bands]
    his = [b.hi for b in p.bands]
    means = [b.mean for b in p.bands]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(mags, los, his, color="tab:blue", alpha=0.25, label="observed band (lo .. hi)")
    ax.plot(mags, means, color="tab:blue", linewidth=2.0, marker="o", markersize=4, label="band mean")
    for k, t in enumerate(p.placement.thresholds):
        ax.hlines(t, k, k + 1, color="tab:orange", linewidth=1.4)
    # Legend proxy for the threshold segments.
    ax.plot([], [], color="tab:orange", linewidth=1.4, label="placed threshold")
    # Unit-glyph LOCAL EXCEPTION to the global units-ASCII convention:
    # matplotlib strings spell units as unicode glyphs (µA) per
    # scientific-figure convention; math variables stay LaTeX.
    ax.set_xlabel(r"per-phase $|M|$")
    ax.set_ylabel(r"$I_{\mathrm{in}}$ [µA]")
    ax.set_title(f"I(M) grid — mode {p.adc_mode} (min margin {p.placement.min_margin:+.4f} µA)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote grid-curve plot to %s", output_path)


def _plot_margins(placements: list[ModePlacement], output_path: Path) -> None:
    """One PNG: per-mode band margins by code boundary (negative = overlap)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    mode_num = len(placements)
    width = 0.8 / mode_num
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(sorted(placements, key=lambda p: p.adc_mode)):
        xs = [k + (i - (mode_num - 1) / 2) * width for k in range(len(p.placement.margins))]
        color = cmap(i / max(mode_num - 1, 1))
        ax.bar(xs, p.placement.margins, width=width, color=color, label=f"mode {p.adc_mode}")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("code boundary k (band k vs k+1)")
    # Unit-glyph LOCAL EXCEPTION (µA), as on the grid-curve plot.
    ax.set_ylabel(r"band margin $lo(k{+}1) - hi(k)$ [µA]")
    ax.set_title("Band margins per mode — negative bars mean adjacent bands overlap")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote margin plot to %s", output_path)


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic macro-ADC threshold probe (config-driven)")
    add_standard_args(parser, output_file=True)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("log/calibration/figures"),
        help="Directory for grid/margin PNGs (default log/calibration/figures)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/calibration"),
        help="Directory for the per-run log file (default log/calibration)",
    )
    parser.add_argument(
        "--element-range",
        type=str,
        default=None,
        help="Contiguous battery-element slice 'a:b' (b empty = to the end) to probe in this run",
    )
    parser.add_argument(
        "--capture-out",
        type=Path,
        default=None,
        help="Save this run's pooled (|M|, i_in__uA) streams to a .pt part file and skip placement",
    )
    parser.add_argument(
        "--capture-in",
        type=str,
        default=None,
        help="Comma-separated .pt part files merged ahead of this run's probed slice",
    )
    return parser


def _parse_element_range(spec: str | None) -> tuple[int, int | None]:
    """Parse ``'a:b'`` (either side optional) into a ``(start, stop)`` slice."""
    if spec is None:
        return (0, None)
    head, sep, tail = spec.partition(":")
    if not sep:
        raise SystemExit(f"--element-range must be 'a:b'; got {spec!r}")
    start = int(head) if head else 0
    stop = int(tail) if tail else None
    return (start, stop)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    log_path = add_file_logging(args.log_dir, "threshold_probe")
    logger.info("log file: %s", log_path)
    device = torch.device(args.device)
    logger.info("device: %s", device)

    cfg = load_tool_config(ThresholdProbeToolConfig, args.config)
    modes_path = resolve_relative_path(cfg.modes_file, args.config)
    assert modes_path is not None
    mode_set = load_mode_set(modes_path)
    # The ladder top magnitude must cover the mode's design range.
    mode_m_max = [(m.adc_mode, math.ceil(m.range)) for m in mode_set.modes]
    logger.info(
        "mode set %s: %s",
        modes_path,
        ", ".join(f"mode {a} m_max {mm}" for a, mm in mode_m_max),
    )
    physical = build_physical_macro(cfg.macro, base=args.config, device=device)
    ideal = build_ideal_twin(physical, device=device)

    grid_top = max(mm for _, mm in mode_m_max)
    magnitude, i_in = _probe_grid(
        physical, ideal, cfg=cfg, grid_top=grid_top, element_range=_parse_element_range(args.element_range)
    )

    if args.capture_out is not None:
        args.capture_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"magnitude": magnitude, "i_in__uA": i_in}, args.capture_out)
        logger.info("wrote capture part (%d samples) to %s — placement deferred", i_in.numel(), args.capture_out)
        return 0

    if args.capture_in is not None:
        m_parts = []
        i_parts = []
        for part in args.capture_in.split(","):
            payload = torch.load(Path(part), map_location="cpu", weights_only=True)
            m_parts.append(payload["magnitude"])
            i_parts.append(payload["i_in__uA"])
            logger.info("merged capture part %s (%d samples)", part, payload["i_in__uA"].numel())
        magnitude = torch.cat([*m_parts, magnitude])
        i_in = torch.cat([*i_parts, i_in])
    logger.info("pooled %d conversion samples", i_in.numel())

    # Grid-curve linearity diagnostic over the pooled samples.
    line = fit_linear(magnitude.to(torch.float64), i_in)
    logger.info(
        "pooled I(M) line fit: slope = %.6f uA/LSB  intercept = %.6f uA  R^2 = %.6f",
        line.slope,
        line.intercept,
        line.r2,
    )

    placements: list[ModePlacement] = []
    for adc_mode, m_max in mode_m_max:
        bands = band_stats(magnitude, i_in, m_max=m_max)
        placement = place_thresholds(bands)
        placements.append(ModePlacement(adc_mode=adc_mode, m_max=m_max, bands=bands, placement=placement))
        _log_mode(placements[-1])

    worst = min(placements, key=lambda p: p.placement.min_margin)
    logger.info(
        "headline: minimum band margin %+f uA (mode %d, boundary %d/%d); monotone all modes = %s",
        worst.placement.min_margin,
        worst.adc_mode,
        worst.placement.min_margin_boundary,
        worst.placement.min_margin_boundary + 1,
        all(p.placement.monotone for p in placements),
    )

    lines = _fragment_lines(placements)
    logger.info("%s", "\n".join(lines))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        logger.info("wrote TOML fragment to %s", args.output)

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for p in placements:
            _plot_grid_curve(p, args.plot_dir / f"threshold_grid_mode{p.adc_mode}.png")
        _plot_margins(placements, args.plot_dir / "threshold_margins.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
