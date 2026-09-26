"""Random characterization and targeted coverage of ADC-input clusters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.tools.calibration.cim_macro.construction import load_macro_config

from ._collection import collect_random_probe_data, collect_targeted_probe_data, ideal_value_counts, plan_target_batches
from ._report import report_probe_data
from .config import AdcProbeToolConfig
from .data import AdcProbeData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdcProbeResult:
    """Paired observations from the random and supplemental phases."""

    random: AdcProbeData
    targeted: AdcProbeData


def characterize_adc(
    config: AdcProbeToolConfig,
    *,
    run_config_path: Path,
    device: torch.device,
    min_samples_per_ideal_value: int,
    on_phase: Callable[[str, AdcProbeData], None] | None = None,
) -> AdcProbeResult:
    """Run both characterization phases and require the requested coverage.

    `on_phase` receives each completed phase before subsequent validation,
    allowing callers to retain observations even if a later phase fails.
    Phase names are `random` and `targeted`.
    """
    if min_samples_per_ideal_value <= 0:
        raise ValueError("min_samples_per_ideal_value must be positive")
    random_data = collect_random_probe_data(config, run_config_path=run_config_path, device=device)
    if on_phase is not None:
        on_phase("random", random_data)

    macro_config = load_macro_config(config.macro, base=run_config_path)
    target_samples_per_batch = config.stimulus.target.batch_w * macro_config.output_num
    target_batch_num = plan_target_batches(
        random_data,
        min_samples_per_ideal_value=min_samples_per_ideal_value,
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
        run_config_path=run_config_path,
        device=device,
    )
    if on_phase is not None:
        on_phase("targeted", targeted_data)

    combined_counts = ideal_value_counts(random_data) + ideal_value_counts(targeted_data)
    minimum_count, minimum_index = combined_counts.min(dim=0)
    minimum_ideal_value = random_data.ideal_value_support[int(minimum_index)]
    if int(minimum_count) < min_samples_per_ideal_value:
        raise ValueError(
            f"target supplementation failed: ideal value {minimum_ideal_value} has {int(minimum_count)} samples, "
            f"below {min_samples_per_ideal_value}"
        )
    logger.info("supplemental paired samples: %d", targeted_data.ideal_value.numel())
    logger.info(
        "minimum combined coverage: ideal_value = %d, count = %d",
        minimum_ideal_value,
        int(minimum_count),
    )
    logger.info("random phase summary:")
    report_probe_data(random_data)
    return AdcProbeResult(random=random_data, targeted=targeted_data)
