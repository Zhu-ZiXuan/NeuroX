"""CLI adapter for full-resolution macro rescale calibration."""

from __future__ import annotations

import argparse

import torch

from neurox.tools.calibration.cli import add_calibration_args
from neurox.tools.calibration.output import emit_fragment
from neurox.tools.calibration.run import calibration_run

from ._report import plot_mode_fit
from .rescale_fit import RescaleFitToolConfig, fit_macro_rescale, rescale_fragment_text


def _modes(value: str) -> tuple[int, ...]:
    try:
        modes = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("modes must be comma-separated integers") from error
    if any(mode < 0 for mode in modes) or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("modes must be distinct nonnegative integers")
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit full-resolution macro rescale factors")
    add_calibration_args(parser, fragment=True, plots=True)
    parser.add_argument(
        "--modes", type=_modes, default=None, help="Comma-separated mode subset; defaults to all modes."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with calibration_run(RescaleFitToolConfig, args, name="rescale_fit") as (config, output):
        result = fit_macro_rescale(
            config, run_config_path=args.config, device=torch.device(args.device), modes=args.modes
        )
        emit_fragment(rescale_fragment_text(result), output, name="rescale.toml", destination=args.output)
        plot_dir = output.plot_dir(args.plot_dir)
        for mode in result.modes:
            plot_mode_fit(mode, plot_dir / f"rescale_fit_mode{mode.quantization_mode}.png")
    return 0
