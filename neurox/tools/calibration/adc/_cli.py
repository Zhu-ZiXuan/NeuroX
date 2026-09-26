"""CLI adapter for ADC-input characterization."""

from __future__ import annotations

import argparse

import torch

from neurox.tools.calibration.cli import add_calibration_args
from neurox.tools.calibration.run import calibration_run
from neurox.tools.cli import positive_int

from .config import AdcProbeToolConfig
from .data import AdcProbeData, save_adc_probe_data
from .run import characterize_adc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe nominal ADC-input distributions for one macro mode")
    add_calibration_args(parser)
    parser.add_argument(
        "--min-samples-per-ideal-value",
        type=positive_int,
        default=65536,
        help="Minimum combined samples for every signed ideal value (default 65536).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with calibration_run(AdcProbeToolConfig, args, name="adc_probe") as (config, output):

        def save_phase(name: str, data: AdcProbeData) -> None:
            save_adc_probe_data(data, output.path(f"{name}.pt"))

        characterize_adc(
            config,
            run_config_path=args.config,
            device=torch.device(args.device),
            min_samples_per_ideal_value=args.min_samples_per_ideal_value,
            on_phase=save_phase,
        )
    return 0
