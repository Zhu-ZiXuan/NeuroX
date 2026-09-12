"""Figures for complete macro rescale fits."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .rescale_fit import ModeFitResult

logger = logging.getLogger(__name__)


def plot_mode_fit(result: ModeFitResult, output_path: Path) -> None:
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
