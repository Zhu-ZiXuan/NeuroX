"""Xue2020 ADC-input distribution plot."""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path

import torch
from torch import Tensor

from neurox.tools.calibrate_adc import AdcProbeData

logger = logging.getLogger(__name__)

_FIGURE_WIDTH_IN = 18.0
_DENSITY_BIN_NUM = 4096
_MIN_SAMPLES_PER_MAGNITUDE = 65536


def _density_edges(value: Tensor, bin_num: int) -> Tensor:
    lower = float(value.min())
    upper = float(value.max())
    if lower == upper:
        padding = max(abs(lower) * 0.01, 0.5)
        lower -= padding
        upper += padding
    return torch.linspace(lower, upper, bin_num + 1, dtype=torch.float64)


def _conditional_density(value: Tensor, edges: Tensor) -> Tensor:
    count = torch.histogram(value.to(torch.float64), bins=edges).hist
    return count / (value.numel() * (edges[1:] - edges[:-1]))


def _contrast_color(index: int) -> tuple[float, float, float]:
    hue = (0.07 + index * 0.6180339887498949) % 1.0
    return colorsys.hls_to_rgb(hue, 0.42, 0.75)


def plot_magnitude_ridgeline(
    data: AdcProbeData,
    output_path: Path,
) -> None:
    """Plot ADC-input distributions after folding signed ideal values."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    magnitude = data.ideal_value.abs()
    magnitude_codes = torch.arange(
        max(abs(value) for value in data.ideal_value_support) + 1,
        dtype=torch.int64,
    )
    edges = _density_edges(data.input_value, _DENSITY_BIN_NUM)
    figure_height = max(12.0, magnitude_codes.numel() * 0.30 + 2.2)
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH_IN, figure_height))
    for index, ideal_magnitude in enumerate(magnitude_codes):
        value = data.input_value[magnitude == ideal_magnitude]
        if value.numel() == 0:
            continue
        density = _conditional_density(value, edges)
        peak = density.max()
        if peak:
            density = density / peak
        baseline = float(index)
        density_values = density.numpy() + baseline
        edge_values = edges.numpy()
        color = _contrast_color(index)
        ax.stairs(
            density_values,
            edge_values,
            baseline=baseline,
            fill=True,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.stairs(
            density_values,
            edge_values,
            baseline=None,
            fill=False,
            color=color,
            linewidth=0.55,
            linestyle="--" if value.numel() < _MIN_SAMPLES_PER_MAGNITUDE else "-",
        )

    ax.set_xlabel(data.input_name)
    ax.set_ylabel("absolute exact ideal value")
    ax.set_title("Xue2020 ADC input distributions by absolute ideal value")
    ax.set_yticks(
        range(magnitude_codes.numel()),
        labels=[str(int(value)) for value in magnitude_codes],
        fontsize=6,
    )
    ax.set_ylim(-0.25, magnitude_codes.numel() - 0.05)
    ax.grid(True, axis="x", alpha=0.20)
    ax.text(
        0.99,
        1.005,
        f"{_DENSITY_BIN_NUM} shared bins; each row independently peak-normalized\n"
        f"signed clusters are folded together; empty rows have no samples; "
        f"dashed rows have n < {_MIN_SAMPLES_PER_MAGNITUDE}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    logger.info("wrote absolute-code ridgeline to %s", output_path)


def plot_cluster_statistics(
    data: AdcProbeData,
    output_path: Path,
    *,
    ideal_boundaries: tuple[int, ...],
) -> None:
    """Plot per-magnitude ADC-input ranges and location statistics."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    magnitude = data.ideal_value.abs()
    max_magnitude = max(abs(value) for value in data.ideal_value_support)
    codes = list(range(max_magnitude + 1))
    observed: list[int] = []
    minimum: list[float] = []
    maximum: list[float] = []
    p05: list[float] = []
    p95: list[float] = []
    mean: list[float] = []
    median: list[float] = []
    probabilities = torch.tensor((0.05, 0.50, 0.95), dtype=torch.float64)
    for code in codes:
        value = data.input_value[magnitude == code].to(torch.float64)
        if value.numel() == 0:
            continue
        quantiles = torch.quantile(value, probabilities)
        observed.append(code)
        minimum.append(float(value.min()))
        maximum.append(float(value.max()))
        p05.append(float(quantiles[0]))
        mean.append(float(value.mean()))
        median.append(float(quantiles[1]))
        p95.append(float(quantiles[2]))

    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH_IN, 7.0))
    ax.fill_between(observed, minimum, maximum, color="tab:blue", alpha=0.12, label="min .. max")
    ax.fill_between(observed, p05, p95, color="tab:blue", alpha=0.30, label="p05 .. p95")
    ax.plot(observed, mean, color="tab:blue", linewidth=1.3, marker="o", markersize=3.0, label="mean")
    ax.plot(observed, median, color="tab:orange", linewidth=1.0, label="median")
    for code, value in zip(observed, mean, strict=True):
        ax.annotate(f"{value:.4g}", (code, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    for boundary in ideal_boundaries:
        ax.axvline(boundary - 0.5, color="black", alpha=0.16, linewidth=0.7, linestyle=":")

    ax.set_xlim(-0.5, max_magnitude + 0.5)
    ax.set_xlabel("absolute exact ideal value")
    ax.set_ylabel(data.input_name)
    ax.set_title("Xue2020 ADC magnitude-cluster statistics")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", framealpha=0.90)
    ax.text(
        0.99,
        1.005,
        "unobserved codes are skipped; dotted lines separate fixed ADC output bins",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    logger.info("wrote magnitude-cluster statistics to %s", output_path)
