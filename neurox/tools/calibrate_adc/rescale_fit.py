"""CLI: LS-fit the per-mode rescale factor of a CIM macro via dual probed runs.

CLI: ``python -m neurox.tools.calibrate_adc.rescale_fit --config <run.toml>
[--device cuda:N] [--output <fragment.toml>] [--plot-dir <dir>]
[--log-dir <dir>] [--log-level INFO] [--modes m[,m...]]``

The operating modes come from the mode-set TOML named by the run config
(``modes_file``, see :mod:`._modes`); ``--modes`` narrows the run to a
subset of that set (each mode re-runs the full stimulus battery, so a
per-mode run bounds single-command runtime; the emitted fragments
concatenate).

For each requested ``quantization_mode`` the tool programs random ternary
weight patterns into the physical tile and its lossless
:meth:`~neurox.primitive.macro.cim.CimMacro.to_ideal` twin, drives random
binary WL batches through both, pairs the physical tile's
``current_adc.convert`` records with the ideal twin's ``vec_mat_mul``
return element for element, maps the ideal dots onto the macro's ADC input
code axis
(:meth:`~neurox.primitive.macro.cim.CimMacro.map_quantization_input_code`),
drops pairs outside that mode's input code range and top-code-saturated
pairs (both drop counts logged per mode), and solves the
zero-through-origin least squares ``ideal_code ~= rescale_factor * code``.
The fit target is the twin's real-valued code scale ``M * 2^B / W``
(``W`` the mode's window width, ``B`` the twin's ``adc_max_bits``), so the
fitted slope is exactly the rescale currency: the macro's code expressed
in ideal-macro codes. The fit runs at the macro's ``adc_max_bits`` only —
lower bit widths follow the base-class law ``r_b = r_B * 2^(B - b)``.
Output is a ``[[modes]]`` macro-config fragment (one table per mode) plus
a per-mode fit plot (code vs ideal code + fitted line).

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
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy, IdealCimMacro
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

    quantization_mode: int
    adc_bits: int
    quantization_input_range: tuple[int, int]
    adc_input_code_range: tuple[int, int]
    fit: RescaleFit
    total_num: int
    range_dropped_num: int
    saturated_num: int
    code: torch.Tensor
    ideal_code: torch.Tensor


def _fit_one_mode(
    physical: CimMacro[CimMacroConfig, CimMacroPolicy],
    ideal: IdealCimMacro,
    *,
    mode: AdcMode,
    adc_bits: int,
    stimulus: _StimulusCfg,
    row_num: int,
    col_num: int,
) -> ModeFitResult:
    """Run the stimulus battery at one mode and solve the rescale.

    The fit target is the ideal twin's real-valued code scale
    ``input_code * 2^B / W``: the unquantized code the twin's window would
    read, so the slope is the physical code expressed in ideal codes.
    """
    quantization_mode = mode.quantization_mode
    window = ideal.quantization_input_ranges[quantization_mode]
    width = window[1] - window[0] + 1
    ideal_code_scale = float(1 << ideal.adc_max_bits) / width
    _, code_range = physical.map_quantization_input_code(
        torch.zeros((), dtype=torch.int64), quantization_mode=quantization_mode
    )

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
                    quantization_mode=quantization_mode,
                    adc_bits=adc_bits,
                )
                input_code, _ = physical.map_quantization_input_code(pair.ideal_m, quantization_mode=quantization_mode)
                code_parts.append(pair.code)
                ideal_parts.append(input_code)
    code = torch.cat(code_parts)
    adc_input_code = torch.cat(ideal_parts)
    # Ideal-axis design-domain filter (the mode only serves input codes its
    # converter resolves) AND top-code-saturation filter (no linear-region
    # information).
    selection = filter_fit_samples(
        code,
        adc_input_code,
        adc_input_code_range=code_range,
        top_code=(1 << adc_bits) - 1,
    )
    keep = selection.keep
    ideal_code = adc_input_code.to(torch.float64) * ideal_code_scale
    fit = fit_rescale_through_origin(code[keep], ideal_code[keep])
    return ModeFitResult(
        quantization_mode=quantization_mode,
        adc_bits=adc_bits,
        quantization_input_range=window,
        adc_input_code_range=code_range,
        fit=fit,
        total_num=int(code.numel()),
        range_dropped_num=selection.range_dropped_num,
        saturated_num=selection.saturated_num,
        code=code[keep],
        ideal_code=ideal_code[keep],
    )


# --- output -----------------------------------------------------------------


def _fragment_lines(results: list[ModeFitResult]) -> list[str]:
    """The ``[[modes]]`` macro-config TOML fragment (nest under the macro section).

    One table per mode, in mode order — the config reads the mode index
    from the table position, so a partial run's tables paste into the
    matching slots.
    """
    lines = [
        "# modes fragment fitted by neurox.tools.calibrate_adc.rescale_fit",
        f"# at adc_bits = {results[0].adc_bits if results else 0} (the macro's adc_max_bits);",
        "# nest each table under the macro config section when pasting",
        "# (e.g. [[cim_macro.modes]]), keeping the mode order.",
    ]
    for r in sorted(results, key=lambda r: r.quantization_mode):
        lower, upper = r.quantization_input_range
        code_lower, code_upper = r.adc_input_code_range
        lines += [
            f"# quantization_mode = {r.quantization_mode}",
            "[[modes]]",
            f"quantization_input_range = [{lower}, {upper}]",
            f"adc_input_code_range = [{code_lower}, {code_upper}]",
            f"max_bits_rescale_factor = {r.fit.rescale_factor:.6f}",
        ]
    return lines


def _plot_mode_fit(result: ModeFitResult, output_path: Path) -> None:
    """One PNG: probed (code, ideal code) scatter + the fitted line."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        result.code.numpy(),
        result.ideal_code.numpy(),
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
    ax.set_xlabel("macro output code")
    ax.set_ylabel("ideal macro code")
    ax.set_title(f"Rescale fit — mode {result.quantization_mode}, {result.adc_bits} bits")
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
        help="Comma-separated quantization_mode subset of the mode set to fit in this run",
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
        known = {mode.quantization_mode for mode in mode_set.modes}
        for mode_idx in selected:
            if mode_idx not in known:
                raise SystemExit(f"--modes entry {mode_idx} not in the mode set {sorted(known)} ({modes_path})")
        modes = tuple(mode for mode in mode_set.modes if mode.quantization_mode in selected)
    else:
        modes = mode_set.modes
    # The fit runs at the macro's max bits; every lower bit width follows
    # the base-class rescale law from the fitted max-bits factor.
    adc_bits = physical.adc_max_bits
    mode_num = len(physical.quantization_input_ranges)
    for mode in modes:
        if not (0 <= mode.quantization_mode < mode_num):
            raise SystemExit(f"mode-set quantization_mode {mode.quantization_mode} outside [0, {mode_num})")
    logger.info(
        "fitting modes %s at adc_bits = %d on %s",
        [mode.quantization_mode for mode in modes],
        adc_bits,
        device,
    )

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
            "mode %d: max_bits_rescale_factor = %.6f  R^2 = %.6f  rmse = %.4f  max|res| = %.4f  "
            "samples = %d of %d (input code outside [%d, %d] excluded %d, top-code-saturated excluded %d)",
            mode.quantization_mode,
            result.fit.rescale_factor,
            result.fit.r2,
            result.fit.rmse,
            result.fit.max_abs_residual,
            result.fit.sample_num,
            result.total_num,
            *result.adc_input_code_range,
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
            _plot_mode_fit(result, args.plot_dir / f"rescale_fit_mode{result.quantization_mode}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
