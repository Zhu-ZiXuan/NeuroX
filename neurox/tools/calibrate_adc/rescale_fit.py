"""CLI: LS-fit the per-(mode, bits) ADC rescale of a CIM macro via dual probed runs.

CLI: ``python -m neurox.tools.calibrate_adc.rescale_fit --config <run.toml>
[--device cuda:N] [--output <fragment.toml>] [--plot-dir <dir>]
[--log-dir <dir>] [--log-level INFO] [--modes m[,m...]]``

The operating modes come from the mode-set TOML named by the run config
(``modes_file``, see :mod:`._modes`); ``--modes`` narrows the run to a
subset of that set (each mode re-runs the full stimulus battery, so a
per-mode run bounds single-command runtime; the emitted fragments
concatenate).

For each requested ``adc_mode`` the tool programs random ternary weight
patterns into the physical tile and its lossless
:meth:`~neurox.primitive.macro.cim.CimMacro.to_ideal` twin, drives random
binary WL batches through both, pairs the physical tile's
``current_adc.convert`` observations with the ideal twin's ``vec_mat_mul``
return element for element, drops pairs outside the
mode's design range on the ideal axis (``|M_ideal| > range``) and
top-code-saturated pairs (both drop counts logged per mode), and solves
the zero-through-origin least squares
``|M_ideal| ~= rescale_factor * code``. Output is an ``[[adc_calibration]]``
TOML fragment (one record per (mode, bits)) plus a per-mode fit plot
(code vs ideal + fitted line).

See also:
    docs/guides/calibration/calibrate_adc.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.common import ConfigBase
from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.macro.cim.ideal import IdealCimMacro
from neurox.tools._config import add_standard_args, load_tool_config, resolve_relative_path, setup_logging

from ._math import RescaleFit, filter_fit_samples, fit_rescale_through_origin
from ._modes import AdcMode, load_mode_set
from ._testbench import (
    MacroSection,
    add_file_logging,
    build_ideal_twin,
    build_physical_macro,
    run_paired_stimulus,
    sample_binary_x,
    sample_ternary_w,
)

logger = logging.getLogger(__name__)


# --- config schema ----------------------------------------------------------


@dataclass(frozen=True)
class _StimulusCfg:
    """``[stimulus]`` section: the random calibration workload.

    Attributes:
        seed: RNG seed for weights and drives.
        w_densities: Ternary non-zero densities; one weight-pattern group
            per density.
        x_densities: Binary WL drive densities crossed with every weight
            pattern.
        patterns_per_density: Independent weight patterns per ``w_density``.
        x_batch: Drive vectors per (pattern, x_density) combo, batched
            into a single VMM call.
    """

    seed: int
    w_densities: tuple[float, ...]
    x_densities: tuple[float, ...]
    patterns_per_density: int
    x_batch: int

    def __post_init__(self) -> None:
        if not self.w_densities or not self.x_densities:
            raise ValueError("require: [stimulus].w_densities and x_densities non-empty")
        for name in ("patterns_per_density", "x_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"require: [stimulus].{name} ({getattr(self, name)}) >= 1")
        for d in (*self.w_densities, *self.x_densities):
            if not (0.0 <= d <= 1.0):
                raise ValueError(f"require: densities in [0, 1]; got {d}")


class RescaleFitToolConfig(ConfigBase):
    """Top-level config for :mod:`neurox.tools.calibrate_adc.rescale_fit`.

    Attributes:
        macro: The tile to build.
        stimulus: The random calibration workload.
        modes_file: Mode-set TOML (see
            :func:`~neurox.tools.calibrate_adc._modes.load_mode_set`),
            relative to the tool TOML; every mode is fitted unless
            ``--modes`` selects a subset.
    """

    macro: MacroSection
    stimulus: _StimulusCfg
    modes_file: Path


# --- fit --------------------------------------------------------------------


@dataclass(frozen=True)
class ModeFitResult:
    """Fit + diagnostics for one operating mode."""

    adc_mode: int
    adc_bits: int
    fit: RescaleFit
    total_num: int
    range_dropped_num: int
    saturated_num: int
    code: torch.Tensor
    ideal_abs: torch.Tensor


def _fit_one_mode(
    physical: CimMacro,
    ideal: IdealCimMacro,
    *,
    mode: AdcMode,
    adc_bits: int,
    stimulus: _StimulusCfg,
    row_num: int,
    col_num: int,
) -> ModeFitResult:
    """Run the stimulus battery at one mode and solve the rescale."""
    gen = torch.Generator().manual_seed(stimulus.seed)
    code_parts: list[torch.Tensor] = []
    ideal_parts: list[torch.Tensor] = []
    for w_density in stimulus.w_densities:
        for _ in range(stimulus.patterns_per_density):
            w = sample_ternary_w(gen, col_num=col_num, row_num=row_num, density=w_density)
            for x_density in stimulus.x_densities:
                x = sample_binary_x(gen, batch=stimulus.x_batch, row_num=row_num, density=x_density)
                pair = run_paired_stimulus(
                    physical,
                    ideal,
                    w=w,
                    x=x,
                    input_num=row_num,
                    adc_mode=mode.adc_mode,
                    adc_bits=adc_bits,
                )
                code_parts.append(pair.code)
                ideal_parts.append(pair.ideal_m.abs())
    code = torch.cat(code_parts)
    ideal_abs = torch.cat(ideal_parts)
    # Ideal-axis design-domain filter (the mode only serves |M| inside its
    # range) AND top-code-saturation filter (no linear-region information).
    selection = filter_fit_samples(code, ideal_abs, range_limit=mode.range, top_code=(1 << adc_bits) - 1)
    keep = selection.keep
    fit = fit_rescale_through_origin(code[keep], ideal_abs[keep])
    return ModeFitResult(
        adc_mode=mode.adc_mode,
        adc_bits=adc_bits,
        fit=fit,
        total_num=int(code.numel()),
        range_dropped_num=selection.range_dropped_num,
        saturated_num=selection.saturated_num,
        code=code[keep],
        ideal_abs=ideal_abs[keep],
    )


# --- output -----------------------------------------------------------------


def _fragment_lines(results: list[ModeFitResult]) -> list[str]:
    """The ``[[adc_calibration]]`` TOML fragment (nest under the macro section)."""
    lines = [
        "# adc_calibration fragment fitted by neurox.tools.calibrate_adc.rescale_fit;",
        "# nest each table under the macro config section when pasting",
        "# (e.g. [[cim_macro.adc_calibration]]).",
    ]
    for r in results:
        lines += [
            "[[adc_calibration]]",
            f"adc_mode = {r.adc_mode}",
            f"adc_bits = {r.adc_bits}",
            f"rescale_factor = {r.fit.rescale_factor:.6f}",
        ]
    return lines


def _plot_mode_fit(result: ModeFitResult, output_path: Path) -> None:
    """One PNG: probed (code, |M_ideal|) scatter + the fitted line."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        result.code.numpy(),
        result.ideal_abs.numpy(),
        s=12,
        alpha=0.35,
        color="tab:blue",
        edgecolors="none",
        label="probed pairs",
    )
    code_max = int(result.code.max()) if result.code.numel() else 1
    grid = torch.arange(0, code_max + 1, dtype=torch.float64)
    ax.plot(
        grid.numpy(),
        (result.fit.rescale_factor * grid).numpy(),
        color="tab:orange",
        linewidth=2.0,
        label=f"fit: rescale = {result.fit.rescale_factor:.4f}  ($R^2$ = {result.fit.r2:.4f})",
    )
    ax.set_xlabel("ADC code")
    ax.set_ylabel(r"$|M_{\mathrm{ideal}}|$")
    ax.set_title(f"ADC rescale fit — mode {result.adc_mode}, {result.adc_bits} bits")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote fit plot to %s", output_path)


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic macro-ADC rescale fit (config-driven)")
    add_standard_args(parser, output_file=True)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("log/calibration/figures"),
        help="Directory for per-mode fit PNGs (default log/calibration/figures)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/calibration"),
        help="Directory for the per-run log file (default log/calibration)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated adc_mode subset of the mode set to fit in this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    log_path = add_file_logging(args.log_dir, "rescale_fit")
    logger.info("log file: %s", log_path)
    device = torch.device(args.device)

    cfg = load_tool_config(RescaleFitToolConfig, args.config)
    modes_path = resolve_relative_path(cfg.modes_file, args.config)
    assert modes_path is not None
    mode_set = load_mode_set(modes_path)
    physical = build_physical_macro(cfg.macro, base=args.config, device=device)
    ideal = build_ideal_twin(physical, device=device)

    if args.modes is not None:
        selected = tuple(int(value) for value in args.modes.split(","))
        known = {mode.adc_mode for mode in mode_set.modes}
        for mode_idx in selected:
            if mode_idx not in known:
                raise SystemExit(f"--modes entry {mode_idx} not in the mode set {sorted(known)} ({modes_path})")
        modes = tuple(mode for mode in mode_set.modes if mode.adc_mode in selected)
    else:
        modes = mode_set.modes
    adc_bits = physical.adc_max_bits
    for mode in modes:
        if not (0 <= mode.adc_mode < physical.adc_mode_num):
            raise SystemExit(f"mode-set adc_mode {mode.adc_mode} outside [0, adc_mode_num ({physical.adc_mode_num}))")
    logger.info("fitting modes %s at adc_bits = %d on %s", [mode.adc_mode for mode in modes], adc_bits, device)

    results: list[ModeFitResult] = []
    for mode in modes:
        result = _fit_one_mode(
            physical,
            ideal,
            mode=mode,
            adc_bits=adc_bits,
            stimulus=cfg.stimulus,
            row_num=cfg.macro.input_num,
            col_num=cfg.macro.output_num,
        )
        results.append(result)
        logger.info(
            "mode %d: rescale_factor = %.6f  R^2 = %.6f  rmse = %.4f  max|res| = %.4f  "
            "samples = %d of %d (out-of-range |M| > %g excluded %d, top-code-saturated excluded %d)",
            mode.adc_mode,
            result.fit.rescale_factor,
            result.fit.r2,
            result.fit.rmse,
            result.fit.max_abs_residual,
            result.fit.sample_num,
            result.total_num,
            mode.range,
            result.range_dropped_num,
            result.saturated_num,
        )

    lines = _fragment_lines(results)
    logger.info("%s", "\n".join(lines))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        logger.info("wrote TOML fragment to %s", args.output)

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            _plot_mode_fit(result, args.plot_dir / f"rescale_fit_mode{result.adc_mode}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
