"""Statistics for stored ADC-input characterization observations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import AdcProbeData

logger = logging.getLogger(__name__)

_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_TORCH_QUANTILE_MAX_NUMEL = 2**24


@dataclass(frozen=True)
class ValueStats:
    count: int
    minimum: float
    p01: float
    p05: float
    p25: float
    mean: float
    median: float
    p75: float
    p95: float
    p99: float
    maximum: float
    std: float


@dataclass(frozen=True)
class ClusterStats:
    ideal_value: int
    signal: ValueStats


@dataclass(frozen=True)
class ProbeSummary:
    global_signal: ValueStats
    clusters: tuple[ClusterStats, ...]
    observed_ideal_range: tuple[int, int]


def _value_stats(value: Tensor) -> ValueStats:
    value = value.detach().flatten().to(torch.float64)
    if value.numel() == 0:
        raise ValueError("require: at least one value")
    quantiles = _quantiles(value)
    return ValueStats(
        count=value.numel(),
        minimum=float(value.min()),
        p01=float(quantiles[0]),
        p05=float(quantiles[1]),
        p25=float(quantiles[2]),
        mean=float(value.mean()),
        median=float(quantiles[3]),
        p75=float(quantiles[4]),
        p95=float(quantiles[5]),
        p99=float(quantiles[6]),
        maximum=float(value.max()),
        std=float(value.std(unbiased=False)),
    )


def _quantiles(value: Tensor) -> Tensor:
    probabilities = torch.tensor(_QUANTILES, dtype=torch.float64)
    if value.numel() <= _TORCH_QUANTILE_MAX_NUMEL:
        return torch.quantile(value, probabilities)

    ordered = value.sort().values
    positions = probabilities * (value.numel() - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    fraction = positions - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_probe(
    ideal_value: Tensor,
    input_value: Tensor,
) -> ProbeSummary:
    """Summarize the input-signal distribution globally and per ideal value."""
    ideal = ideal_value.detach().flatten().long()
    signal = input_value.detach().flatten().to(torch.float64)
    if ideal.numel() != signal.numel():
        raise ValueError(f"require: ideal_value numel ({ideal.numel()}) == input_value numel ({signal.numel()})")
    if ideal.numel() == 0:
        raise ValueError("require: at least one paired ADC sample")

    observed = tuple(int(value) for value in torch.unique(ideal, sorted=True).tolist())
    clusters = tuple(
        ClusterStats(
            ideal_value=value,
            signal=_value_stats(signal[ideal == value]),
        )
        for value in observed
    )
    return ProbeSummary(
        global_signal=_value_stats(signal),
        clusters=clusters,
        observed_ideal_range=(observed[0], observed[-1]),
    )


def _log_summary(result: AdcProbeData, summary: ProbeSummary) -> None:
    global_signal = summary.global_signal
    logger.info("input signal: %s", result.input_name)
    logger.info("paired samples: %d", global_signal.count)
    logger.info("theoretical ideal range: %d .. %d", *result.ideal_value_range)
    logger.info("reachable signed ideal values: %d", len(result.ideal_value_support))
    logger.info("observed ideal range: %d .. %d", *summary.observed_ideal_range)
    logger.info("observed clusters: %d", len(summary.clusters))
    smallest = min(summary.clusters, key=lambda cluster: cluster.signal.count)
    logger.info(
        "least-sampled cluster: ideal_value = %d, count = %d",
        smallest.ideal_value,
        smallest.signal.count,
    )
    logger.info(
        "global %s: min %.8g  p01 %.8g  p05 %.8g  p25 %.8g  mean %.8g  "
        "median %.8g  p75 %.8g  p95 %.8g  p99 %.8g  max %.8g  std %.8g",
        result.input_name,
        global_signal.minimum,
        global_signal.p01,
        global_signal.p05,
        global_signal.p25,
        global_signal.mean,
        global_signal.median,
        global_signal.p75,
        global_signal.p95,
        global_signal.p99,
        global_signal.maximum,
        global_signal.std,
    )


def report_probe_data(data: AdcProbeData) -> None:
    """Log distribution statistics for one ADC-probe artifact."""
    summary = summarize_probe(data.ideal_value, data.input_value)
    _log_summary(data, summary)
