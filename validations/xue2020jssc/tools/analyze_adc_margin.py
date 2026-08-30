"""Estimate Xue2020 ADC references between adjacent ideal-value clusters."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from neurox.tools.calibrate_adc import AdcProbeData, load_adc_probe_data
from validations.xue2020jssc.tools._adc_plot import plot_cluster_statistics, plot_magnitude_ridgeline

logger = logging.getLogger(__name__)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_IDEAL_BOUNDARIES = (10, 20, 30, 40, 50, 60, 70)
_TAIL_QUANTILES = (0.95, 0.99)
_PRIMARY_TAIL_QUANTILE = 0.99


@dataclass(frozen=True)
class BoundaryEstimate:
    ideal_boundary: int
    tail_quantile: float
    lower_count: int
    upper_count: int
    lower_edge: float
    upper_edge: float
    margin: float
    midpoint: float


def _quantile(value: Tensor, probability: float) -> float:
    if value.numel() == 0:
        return math.nan
    return float(torch.quantile(value.to(torch.float64), probability))


def estimate_boundaries(
    ideal_value: Tensor,
    input_value: Tensor,
    *,
    ideal_boundaries: tuple[int, ...] = _IDEAL_BOUNDARIES,
    tail_quantiles: tuple[float, ...] = _TAIL_QUANTILES,
) -> tuple[BoundaryEstimate, ...]:
    """Estimate the fixed magnitude references at every inspected tail quantile."""
    ideal = ideal_value.detach().flatten().to(torch.int64).abs()
    signal = input_value.detach().flatten()
    if ideal.numel() != signal.numel():
        raise ValueError(f"require: ideal_value numel ({ideal.numel()}) == input_value numel ({signal.numel()})")

    estimates: list[BoundaryEstimate] = []
    for boundary in ideal_boundaries:
        lower = signal[ideal == boundary - 1]
        upper = signal[ideal == boundary]
        for quantile in tail_quantiles:
            lower_edge = _quantile(lower, quantile)
            upper_edge = _quantile(upper, 1.0 - quantile)
            estimates.append(
                BoundaryEstimate(
                    ideal_boundary=boundary,
                    tail_quantile=quantile,
                    lower_count=lower.numel(),
                    upper_count=upper.numel(),
                    lower_edge=lower_edge,
                    upper_edge=upper_edge,
                    margin=upper_edge - lower_edge,
                    midpoint=(lower_edge + upper_edge) / 2.0,
                )
            )
    return tuple(estimates)


def _merge_probe_data(parts: tuple[AdcProbeData, ...]) -> AdcProbeData:
    first = parts[0]
    for part in parts[1:]:
        if part.input_name != first.input_name:
            raise ValueError(f"ADC input quantities differ: {first.input_name!r} vs {part.input_name!r}")
        if part.ideal_value_support != first.ideal_value_support:
            raise ValueError(
                f"reachable ideal values differ: {first.ideal_value_support} vs {part.ideal_value_support}"
            )
    return AdcProbeData(
        input_name=first.input_name,
        input_value=torch.cat([part.input_value for part in parts]),
        ideal_value=torch.cat([part.ideal_value for part in parts]),
        ideal_value_support=first.ideal_value_support,
    )


def _log_estimates(
    estimates: tuple[BoundaryEstimate, ...],
    *,
    input_name: str,
) -> None:
    logger.info("ADC input quantity: %s", input_name)
    for quantile in sorted({estimate.tail_quantile for estimate in estimates}):
        logger.info("=" * 96)
        logger.info(
            "tail quantile q=%.5g: lower edge=p%.3g, upper edge=p%.3g",
            quantile,
            100 * quantile,
            100 * (1.0 - quantile),
        )
        logger.info(
            "%-10s %12s %12s %14s %14s %14s %14s",
            "boundary",
            "n_lower",
            "n_upper",
            "lower_edge",
            "upper_edge",
            "margin",
            "midpoint",
        )
        for estimate in estimates:
            if estimate.tail_quantile != quantile:
                continue
            logger.info(
                "%-10d %12d %12d %14.8g %14.8g %14.8g %14.8g",
                estimate.ideal_boundary,
                estimate.lower_count,
                estimate.upper_count,
                estimate.lower_edge,
                estimate.upper_edge,
                estimate.margin,
                estimate.midpoint,
            )
    logger.info("=" * 96)
    primary = [estimate for estimate in estimates if estimate.tail_quantile == _PRIMARY_TAIL_QUANTILE]
    logger.info(
        "primary q=%.5g candidate references: [%s]",
        _PRIMARY_TAIL_QUANTILE,
        ", ".join(f"{estimate.midpoint:.8g}" for estimate in primary),
    )
    missing = [estimate.ideal_boundary for estimate in primary if not math.isfinite(estimate.margin)]
    overlapping = [estimate.ideal_boundary for estimate in primary if estimate.margin <= 0.0]
    if missing:
        logger.warning("primary boundaries with missing adjacent clusters: %s", missing)
    if overlapping:
        logger.warning("primary boundaries with overlapping tails: %s", overlapping)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate ADC references from saved probe clusters")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Paired-sample .pt file from calibrate_adc; repeat to merge files",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/calibration"),
        help="Directory for the timestamped analysis log",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for the merged magnitude SVG plots; defaults to <log-dir>/figures",
    )
    parser.add_argument("--log-level", type=str.upper, default="INFO", choices=_LOG_LEVELS)
    return parser


def _setup_logging(level_name: str, log_dir: Path) -> Path:
    logging.basicConfig(level=getattr(logging, level_name), format="%(message)s", force=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"adc_margin_{stamp}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    return log_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    log_path = _setup_logging(args.log_level, args.log_dir)
    logger.info("log file: %s", log_path)
    for path in args.input:
        logger.info("paired samples: %s", path)
    result = _merge_probe_data(tuple(load_adc_probe_data(path) for path in args.input))
    logger.info("merged paired samples: %d", result.input_value.numel())
    estimates = estimate_boundaries(result.ideal_value, result.input_value)
    _log_estimates(estimates, input_name=result.input_name)
    plot_dir = args.plot_dir if args.plot_dir is not None else args.log_dir / "figures"
    plot_magnitude_ridgeline(result, plot_dir / "adc_magnitude_ridgeline.svg")
    plot_cluster_statistics(
        result,
        plot_dir / "adc_cluster_statistics.svg",
        ideal_boundaries=_IDEAL_BOUNDARIES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
