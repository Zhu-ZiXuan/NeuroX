"""Collect and report ADC-input distributions for one CIM-macro mode.

CLI: `python -m neurox.tools.calibrate_adc --config <run.toml>
[--device cuda:N] [--output-dir <dir>] [--min-samples-per-ideal-value 65536]`

See Also:
    docs/guides/calibration/calibrate_adc.md
"""

from __future__ import annotations

import logging

import torch

from neurox.tools._config import load_tool_config, setup_logging
from neurox.tools._logging import attach_file_logging, create_run_directory
from neurox.tools._macro import load_macro_config

from ._cli import parse_args
from ._collection import (
    collect_random_probe_data,
    collect_targeted_probe_data,
    ideal_value_counts,
    plan_target_batches,
)
from ._report import report_probe_data
from ._schema import AdcProbeToolConfig
from .data import save_adc_probe_data

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    run_dir = create_run_directory(args.output_dir, "adc_probe")
    log_path = attach_file_logging(run_dir / "adc_probe.log")
    logger.info("run directory: %s", run_dir)
    logger.info("log file: %s", log_path)
    logger.info("device: %s", args.device)
    logger.info("minimum samples per signed ideal value: %d", args.min_samples_per_ideal_value)
    config = load_tool_config(AdcProbeToolConfig, args.config)
    device = torch.device(args.device)

    random_data = collect_random_probe_data(config, run_config_path=args.config, device=device)
    random_path = run_dir / "random.pt"
    save_adc_probe_data(random_data, random_path)
    logger.info("wrote random samples to %s", random_path)

    macro_config = load_macro_config(config.macro, base=args.config)
    target_samples_per_batch = config.stimulus.target.batch_w * macro_config.output_num
    target_batch_num = plan_target_batches(
        random_data,
        min_samples_per_ideal_value=args.min_samples_per_ideal_value,
        samples_per_batch=target_samples_per_batch,
    )
    logger.info(
        "signed ideal values below threshold after random sampling: %d/%d",
        len(target_batch_num),
        len(random_data.ideal_value_support),
    )
    logger.info("planned supplemental target batches: %d", sum(target_batch_num.values()))
    targeted_data = collect_targeted_probe_data(
        config,
        random_data,
        target_batch_num,
        run_config_path=args.config,
        device=device,
    )
    targeted_path = run_dir / "targeted.pt"
    save_adc_probe_data(targeted_data, targeted_path)
    logger.info("wrote targeted samples to %s", targeted_path)

    combined_counts = ideal_value_counts(random_data) + ideal_value_counts(targeted_data)
    minimum_count, minimum_index = combined_counts.min(dim=0)
    minimum_ideal_value = random_data.ideal_value_support[int(minimum_index)]
    if int(minimum_count) < args.min_samples_per_ideal_value:
        raise ValueError(
            f"target supplementation failed: ideal value {minimum_ideal_value} has {int(minimum_count)} samples, "
            f"below {args.min_samples_per_ideal_value}"
        )
    logger.info("supplemental paired samples: %d", targeted_data.ideal_value.numel())
    logger.info(
        "minimum combined coverage: ideal_value = %d, count = %d",
        minimum_ideal_value,
        int(minimum_count),
    )
    logger.info("random phase summary:")
    report_probe_data(random_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
