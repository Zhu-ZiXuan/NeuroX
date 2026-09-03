"""Fit the per-mode rescale factor of a CIM macro at `adc_bits`.

CLI: `python -m neurox.tools.calibrate_macro.rescale_fit --config <run.toml>
[--device cuda:N] [--output <fragment.toml>] [--plot-dir <dir>]
[--log-dir <dir>] [--log-level INFO] [--modes m[,m...]]`

`--modes` narrows the run to a subset of the macro's configured reference
operating points. Each mode re-runs the full stimulus battery.

Logical weight and input batches are sampled from a configured distribution.
The physical macro and its ideal twin receive identical programs and legal
caller-side active-position planes. The physical result at `adc_bits` and the
twin's highest-precision result already share the logical output layout,
so the fit depends on no ADC implementation or probe. The zero-through-origin
least-squares slope expresses one final full-resolution macro output code in
MAC units.

See Also:
    docs/guides/calibration/calibrate_macro.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.common import ConfigBase, TensorDataClassBase, ValidateMixin
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy, IdealCimMacro
from neurox.tools._config import add_standard_args, load_tool_config, resolve_relative_path, setup_logging
from neurox.tools._logging import add_file_logging
from neurox.tools._macro import MacroSection, build_ideal_twin, build_physical_macro, unroll_active_positions
from neurox.tools._sampling import load_distribution, make_generator, sample_w, sample_x_batches

from ._math import RescaleFit, fit_rescale_through_origin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StimulusCfg(ValidateMixin):
    seed: int
    distribution_file: Path
    weight_samples: int
    input_samples_per_weight: int
    batch_w: int
    batch_x: int

    def __post_init__(self) -> None:
        self._require_pos(self.weight_samples, "[stimulus].weight_samples")
        self._require_pos(self.input_samples_per_weight, "[stimulus].input_samples_per_weight")
        self._require_pos(self.batch_w, "[stimulus].batch_w")
        self._require_pos(self.batch_x, "[stimulus].batch_x")
        if self.weight_samples % self.batch_w:
            raise ValueError("require: weight_samples is divisible by batch_w")
        if self.input_samples_per_weight % self.batch_x:
            raise ValueError("require: input_samples_per_weight is divisible by batch_x")


class RescaleFitToolConfig(ConfigBase):
    macro: MacroSection
    stimulus: _StimulusCfg


class ModeFitResult(TensorDataClassBase):
    """Full-resolution fit and diagnostics for one operating mode."""

    quantization_mode: int
    adc_bits: int
    fit: RescaleFit
    code: torch.Tensor
    """Macro output code entering the fit.
    Shape: `[sample_num]`."""
    ideal_value: torch.Tensor
    """Lossless ideal-macro value of the same pairs.
    Shape: `[sample_num]`."""


def _fit_one_mode(
    physical: CimMacro[CimMacroConfig, CimMacroPolicy],
    ideal: IdealCimMacro,
    *,
    quantization_mode: int,
    adc_bits: int,
    stimulus: _StimulusCfg,
    run_config_path: Path,
    device: torch.device,
) -> ModeFitResult:
    """Fit ideal value from one mode's full-resolution macro code."""
    distribution_path = resolve_relative_path(stimulus.distribution_file, run_config_path)
    distribution = load_distribution(distribution_path, physical)
    generator = make_generator(stimulus.seed, torch.device("cpu"))
    code_parts: list[torch.Tensor] = []
    ideal_parts: list[torch.Tensor] = []
    for w in sample_w(
        distribution,
        physical,
        input_num=physical.input_num,
        output_num=physical.output_num,
        n=stimulus.weight_samples,
        batch_w=stimulus.batch_w,
        device=torch.device("cpu"),
        generator=generator,
    ):
        w = w.to(device)
        physical.program(w)
        ideal.program(w)
        for x in sample_x_batches(
            distribution,
            physical,
            input_num=physical.input_num,
            n_total=stimulus.input_samples_per_weight,
            batch_size=stimulus.batch_x,
            device=torch.device("cpu"),
            generator=generator,
        ):
            x = x.unsqueeze(1).to(device)
            planes = unroll_active_positions(
                x,
                input_num=physical.input_num,
                max_active_num=physical.max_active_num,
                inst_shape=physical.inst_shape,
            )
            with torch.no_grad():
                code = physical.vec_mat_mul(
                    planes,
                    quantization_mode=quantization_mode,
                    adc_active_bits=None,
                )
                ideal_value = ideal.vec_mat_mul(
                    planes,
                    quantization_mode=quantization_mode,
                    adc_active_bits=None,
                )
            code_parts.append(code.flatten().to("cpu", torch.float64))
            ideal_parts.append(ideal_value.flatten().to("cpu", torch.float64))
    code = torch.cat(code_parts)
    ideal_value = torch.cat(ideal_parts)
    fit = fit_rescale_through_origin(code, ideal_value)
    return ModeFitResult(
        quantization_mode=quantization_mode,
        adc_bits=adc_bits,
        fit=fit,
        code=code,
        ideal_value=ideal_value,
    )


def _fragment_lines(results: list[ModeFitResult], factors: tuple[float, ...]) -> list[str]:
    """Return one complete rescale-factor assignment."""
    updated = list(factors)
    for result in results:
        updated[result.quantization_mode] = result.fit.rescale_factor
    lines = [
        "# fitted by neurox.tools.calibrate_macro.rescale_fit",
        f"# at full ADC resolution: adc_bits = {results[0].adc_bits if results else 0};",
        "# tuple position is quantization_mode",
        "rescale_factors = [",
    ]
    lines.extend(f"    {factor:.6f}," for factor in updated)
    return [*lines, "]"]


def _plot_mode_fit(result: ModeFitResult, output_path: Path) -> None:
    """Write one PNG: probed (code, ideal code) scatter + the fitted line."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        result.code.numpy(),
        result.ideal_value.numpy(),
        s=12,
        alpha=0.35,
        color="tab:blue",
        edgecolors="none",
        label="probed pairs",
    )
    code_min = int(result.code.min()) if result.code.numel() else 0
    code_max = int(result.code.max()) if result.code.numel() else 1
    grid = torch.arange(code_min, code_max + 1, dtype=torch.float64)
    ax.plot(
        grid.numpy(),
        (result.fit.rescale_factor * grid).numpy(),
        color="tab:orange",
        linewidth=2.0,
        label=(f"fit: rescale = {result.fit.rescale_factor:.4f}  ($R^2$ = {result.fit.r2:.4f})"),
    )
    ax.set_xlabel("macro output code")
    ax.set_ylabel("ideal macro value")
    ax.set_title(f"Full-resolution rescale fit — mode {result.quantization_mode}, adc_bits = {result.adc_bits}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("wrote fit plot to %s", output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-resolution macro rescale fit (config-driven)")
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
        help="Comma-separated quantization_mode subset to fit in this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    log_path = add_file_logging(args.log_dir, "rescale_fit")
    logger.info("log file: %s", log_path)
    device = torch.device(args.device)

    cfg = load_tool_config(RescaleFitToolConfig, args.config)
    physical = build_physical_macro(
        cfg.macro,
        base=args.config,
        device=device,
        inst_shape=(cfg.stimulus.batch_w,),
    )
    ideal = build_ideal_twin(physical, device=device)

    if args.modes is not None:
        selected = tuple(int(value) for value in args.modes.split(","))
        known = set(range(len(physical.config.rescale_factors)))
        for mode_idx in selected:
            if mode_idx not in known:
                raise SystemExit(f"--modes entry {mode_idx} outside {sorted(known)}")
        modes = selected
    else:
        modes = tuple(range(len(physical.config.rescale_factors)))
    # The fit runs at full ADC resolution; every lower active width follows
    # the base-class rescale law from this factor.
    adc_bits = physical.adc_bits
    logger.info(
        "fitting modes %s at full ADC resolution (adc_bits = %d) on %s",
        list(modes),
        adc_bits,
        device,
    )

    results: list[ModeFitResult] = []
    for quantization_mode in modes:
        result = _fit_one_mode(
            physical,
            ideal,
            quantization_mode=quantization_mode,
            adc_bits=adc_bits,
            stimulus=cfg.stimulus,
            run_config_path=args.config,
            device=device,
        )
        results.append(result)
        logger.info(
            "mode %d: rescale_factor = %.6f  R^2 = %.6f  rmse = %.4f  max|res| = %.4f  samples = %d",
            quantization_mode,
            result.fit.rescale_factor,
            result.fit.r2,
            result.fit.rmse,
            result.fit.max_abs_residual,
            result.fit.sample_num,
        )

    lines = _fragment_lines(results, physical.config.rescale_factors)
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
