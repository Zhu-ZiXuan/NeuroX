"""CLI: probe the ADC analog-input distribution and recommend input-range candidates.

Config-driven. The TOML pulls in a chip xbar (``[xbar]`` section with
``_neurox_use``) plus per-run workload / statistic / plot settings.
Histogram bin widths are auto-derived from the ADC code grid so the
overlay never goes coarser than the code spacing.

See also:
    docs/guides/calibration/README.md
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.tools._config import (
    add_standard_args,
    load_tool_config,
    resolve_relative_path,
    setup_logging,
)
from neurox.tools.xbar_adc._probe import install_probe_adc
from neurox.tools.xbar_adc._sampling import (
    build_offset_1t1r_xbar_all_off_from_config,
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)
from neurox.xbar import Offset1T1RXbarConfig

# ---------------------------------------------------------------------------
# TOML config schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkloadCfg:
    """``[workload]`` section: sampling sweep dimensions + RNG."""

    weight_samples: int
    input_samples_per_weight: int
    batch_size: int
    seed: int
    distribution: Path | None = None


@dataclass(frozen=True)
class _StatisticCfg:
    """``[statistic]`` section: candidate-ladder depth."""

    max_clip_rate_exp: int


@dataclass(frozen=True)
class _PlotCfg:
    """``[plot]`` section: ADC code grid binning controls.

    ``bits`` is the ADC width whose code grid drives the spotlight overlay.
    ``bins_per_code`` sub-divides the overview bin width — bins_per_code = 1
    matches one code per bin; larger oversample shows sub-code structure.
    """

    bits: int
    bins_per_code: int


@dataclass(frozen=True)
class XbarAdcStatisticConfig:
    """Top-level config for :mod:`neurox.tools.xbar_adc.statistic`.

    The ``xbar`` field is typically populated from a separate chip preset
    via ``_neurox_use`` so the run config and circuit config evolve
    independently.
    """

    xbar: Offset1T1RXbarConfig
    workload: _WorkloadCfg
    statistic: _StatisticCfg
    plot: _PlotCfg


logger = logging.getLogger(__name__)

_DUMMY_OP = AdcOperationPoint(adc_mode=0, adc_bits=1)


@dataclass(frozen=True)
class RangeCandidate:
    """One ADC input-range candidate ``[-A, A]`` and its observed clip rate.

    The integer ``clip_rate_exp`` is the single source of truth; all
    presentation forms (label, nominal_clip_rate, filename) are derived
    deterministically from it, avoiding any floating-point round-trip.

    Attributes:
        clip_rate_exp: ``None`` for the ``max_abs`` candidate; otherwise
            the positive integer ``N`` such that nominal clip rate is
            ``10**-N`` (``N=2`` → ``p99``, ``N=3`` → ``p99.9``, ...).
        a__V: Half-range value [V] — full range is ``[-a__V, +a__V]``.
        observed_clip_rate: Empirical fraction of ``|v_diff|`` exceeding ``a__V``.
    """

    clip_rate_exp: int | None
    a__V: float
    observed_clip_rate: float

    @property
    def label(self) -> str:
        """Human-readable label for logs and plot titles."""
        if self.clip_rate_exp is None:
            return "max_abs"
        n = self.clip_rate_exp
        return "p99" if n == 2 else f"p99.{'9' * (n - 2)}"

    @property
    def nominal_clip_rate(self) -> float:
        """Target clip rate — ``0.0`` for ``max_abs``; else ``10**-N``."""
        return 0.0 if self.clip_rate_exp is None else 10.0**-self.clip_rate_exp

    @property
    def filename(self) -> str:
        """Filesystem-safe spotlight filename — uses integer exponent."""
        if self.clip_rate_exp is None:
            return "spotlight_max_abs.png"
        return f"spotlight_exp{self.clip_rate_exp}.png"


@dataclass(frozen=True)
class Statistics:
    """Numerical summary of one statistic-tool run.

    Attributes:
        v_pos__V / v_neg__V / v_diff__V: Captured CPU float64 1-D tensors.
        candidates: Range candidates ordered ``max_abs`` first, then
            ``p99 → p99.9 → ...`` according to ``max_clip_rate_exp``.
        weight_samples / input_samples_per_weight / captured_count:
            Sampling metadata.
        adc_instance_count: Number of physically-identical ``bl_adc``
            instances in the xbar; captured inputs are pooled across all
            of them by design (identical circuits sharing the same bias).
        distribution_source: ``"uniform"`` or the distribution-TOML path.
    """

    v_pos__V: Tensor
    v_neg__V: Tensor
    v_diff__V: Tensor
    candidates: tuple[RangeCandidate, ...]
    weight_samples: int
    input_samples_per_weight: int
    captured_count: int
    adc_instance_count: int
    distribution_source: str


# ---------------------------------------------------------------------------
# Range-candidate ladder
# ---------------------------------------------------------------------------


def build_candidates(
    abs_v_diff: Tensor,
    max_clip_rate_exp: int,
) -> tuple[RangeCandidate, ...]:
    """Build the range-candidate ladder from observed ``|v_diff|``.

    Always emits ``max_abs`` plus one entry per power-of-10 clip rate
    from ``1e-2`` down to ``10**-max_clip_rate_exp``. The integer
    exponent is the sole driver — no floating-point round-trip enters
    any label or filename.

    Args:
        abs_v_diff: 1-D float64 CPU tensor of ``|v_diff|`` samples.
        max_clip_rate_exp: Bottom of the ladder; ``N`` in ``10**-N``.

    Raises:
        ValueError: When ``max_clip_rate_exp < 2``, when no samples were
            captured, or when the captured sample count is insufficient
            to estimate the bottom percentile (``count < 10**(N + 1)``).
    """
    if max_clip_rate_exp < 2:
        raise ValueError(f"max_clip_rate_exp ({max_clip_rate_exp}) must be >= 2")

    sample_count = abs_v_diff.numel()
    if sample_count == 0:
        raise ValueError("no ADC inputs captured; cannot build range candidates")

    # Smallest nominal rate is ``10**-max_clip_rate_exp``; rule of thumb
    # needs >= 10 samples in the tail beyond that rate.
    required = 10 ** (max_clip_rate_exp + 1)
    if sample_count < required:
        smallest_label = "p99" if max_clip_rate_exp == 2 else f"p99.{'9' * (max_clip_rate_exp - 2)}"
        raise ValueError(
            f"insufficient samples for {smallest_label} estimate: "
            f"captured={sample_count}, need >= {required} (rule of thumb: 10/clip_rate). "
            f"Either lower --max-clip-rate-exp ({max_clip_rate_exp}) "
            "or increase [workload].weight_samples and/or [workload].input_samples_per_weight."
        )

    out: list[RangeCandidate] = []
    a_max = float(abs_v_diff.max().item())
    out.append(
        RangeCandidate(
            clip_rate_exp=None,
            a__V=a_max,
            observed_clip_rate=0.0,
        )
    )

    sorted_abs, _ = torch.sort(abs_v_diff)
    n_total = sorted_abs.numel()
    for n in range(2, max_clip_rate_exp + 1):
        rate = 10.0**-n
        # torch.quantile() is capped at 2**24 elements; for larger tensors
        # use direct index lookup into the pre-sorted array (exact, faster).
        if n_total > (1 << 24):
            idx = round((1.0 - rate) * (n_total - 1))
            idx = max(0, min(idx, n_total - 1))
            a = float(sorted_abs[idx].item())
        else:
            a = float(torch.quantile(sorted_abs, 1.0 - rate, interpolation="linear").item())
        observed = float((abs_v_diff > a).to(torch.float64).mean().item())
        out.append(
            RangeCandidate(
                clip_rate_exp=n,
                a__V=a,
                observed_clip_rate=observed,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------


def collect_statistics(
    *,
    xbar_config: Offset1T1RXbarConfig,
    distribution_path: Path | None,
    weight_samples: int,
    input_samples_per_weight: int,
    batch_size: int,
    max_clip_rate_exp: int,
    seed: int | None,
    device: torch.device,
) -> Statistics:
    """Run the full sampling sweep and assemble a :class:`Statistics`.

    Args:
        xbar_config: already-loaded chip xbar config.
        distribution_path: distribution TOML; ``None`` means uniform.
        weight_samples: total number of independent programmed weights.
            Must be a multiple of ``batch_size`` — the quotient is the
            serial program / re-program count.
        input_samples_per_weight: per-programming input vector count.
            Runs in a single VMM call per programming (broadcast across
            the ``batch_size`` parallel weights — not chunked).
        batch_size: parallel weight-programming axis; the xbar is built
            with ``inst_shape=(batch_size,)`` so each VMM evaluates that
            many independent weight programs concurrently.
        max_clip_rate_exp: bottom of the candidate ladder
            (``10**-max_clip_rate_exp``).
        seed: optional generator seed.
        device: target torch device.

    Raises:
        ValueError: from :func:`build_candidates` when sample count is
            below the rule-of-thumb threshold.
    """
    if weight_samples <= 0:
        raise ValueError(f"weight_samples ({weight_samples}) must be > 0")
    if input_samples_per_weight <= 0:
        raise ValueError(f"input_samples_per_weight ({input_samples_per_weight}) must be > 0")
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > 0")
    if weight_samples % batch_size != 0:
        raise ValueError(
            f"weight_samples ({weight_samples}) must be a multiple of batch_size ({batch_size}) "
            f"so the w-batch axis cleanly divides the requested w count"
        )

    # batch_size now = parallel w-batch axis (xbar built with inst_shape=(B,)).
    # input_samples_per_weight = K = inputs per w sample, broadcast across the
    # B parallel weights inside a single VMM call. The two axes are orthogonal:
    # raising batch_size doesn't shrink K, it just packs more independent w
    # programmings per VMM. Total samples = weight_samples · K · n_data_cols.
    xbar = build_offset_1t1r_xbar_all_off_from_config(xbar_config, device=device, inst_shape=(batch_size,))
    distribution = load_distribution(distribution_path, xbar)
    generator = make_generator(seed, device)
    w_groups = weight_samples // batch_size

    logger.info("xbar built on device=%s; distribution_source=%s", device, distribution.source)
    logger.info(
        "sweep: weight_samples=%d (w_batch=%d × %d groups), input_samples_per_weight=%d, max_clip_rate_exp=%d",
        weight_samples,
        batch_size,
        w_groups,
        input_samples_per_weight,
        max_clip_rate_exp,
    )

    with install_probe_adc(xbar) as handle:
        for i, w in enumerate(
            sample_w(distribution, xbar, n=weight_samples, batch_w=batch_size, device=device, generator=generator)
        ):
            xbar.program(w)
            # Single VMM per group: K inputs broadcast against B parallel weights.
            # sample_x_batches yields (K, row_num); reshape to (K, 1, row_num) so
            # the xbar's (B, ...) instance axis is broadcastable.
            for x in sample_x_batches(
                distribution,
                xbar,
                n_total=input_samples_per_weight,
                batch_size=input_samples_per_weight,
                device=device,
                generator=generator,
            ):
                # x.unsqueeze(-2) inserts the weight-instance broadcast
                # slot so input_samples × batch_size(weights) form a
                # Cartesian product inside the xbar's leading broadcast.
                x_with_weight_inst_slot = x.unsqueeze(-2)
                xbar.vec_mat_mul(x_with_weight_inst_slot, adc_operation_point=_DUMMY_OP)
            logger.info(
                "w-group %d/%d done (%d weights, %d inputs each)", i + 1, w_groups, batch_size, input_samples_per_weight
            )

        v_pos, v_neg, v_diff = handle.probe.captured()
        adc_instance_count = int(math.prod(handle.probe.inst_shape))

    candidates = build_candidates(v_diff.abs(), max_clip_rate_exp)
    return Statistics(
        v_pos__V=v_pos,
        v_neg__V=v_neg,
        v_diff__V=v_diff,
        candidates=candidates,
        weight_samples=weight_samples,
        input_samples_per_weight=input_samples_per_weight,
        captured_count=int(v_diff.numel()),
        adc_instance_count=adc_instance_count,
        distribution_source=distribution.source,
    )


# ---------------------------------------------------------------------------
# Logger output + plot
# ---------------------------------------------------------------------------


def _percentile(t: Tensor, q: float) -> float:
    return float(torch.quantile(t, q, interpolation="linear").item())


def log_statistics(stats: Statistics) -> None:
    """Emit the structured per-tensor summary + range-candidate ladder."""
    logger.info("ADC input statistics and range recommendation")
    logger.info("  sample_source: synthetic")
    logger.info("  distribution: %s", stats.distribution_source)
    logger.info("  weight_samples: %d", stats.weight_samples)
    logger.info("  input_samples_per_weight: %d", stats.input_samples_per_weight)
    logger.info("  captured_adc_values: %d", stats.captured_count)
    logger.info(
        "  adc_instance_count: %d (pooled across all instances; identical circuits + shared bias)",
        stats.adc_instance_count,
    )
    logger.info("")

    for name, tensor in (
        ("v_pos__V", stats.v_pos__V),
        ("v_neg__V", stats.v_neg__V),
        ("v_diff__V", stats.v_diff__V),
    ):
        logger.info("%s:", name)
        logger.info("  min: %.6g", float(tensor.min().item()))
        logger.info("  max: %.6g", float(tensor.max().item()))
        logger.info("  mean: %.6g", float(tensor.mean().item()))
        logger.info("  std: %.6g", float(tensor.std(unbiased=False).item()))
        logger.info("  p50: %.6g", _percentile(tensor, 0.50))
        logger.info("  p90: %.6g", _percentile(tensor, 0.90))
        logger.info("  p99: %.6g", _percentile(tensor, 0.99))
        logger.info("  p99.9: %.6g", _percentile(tensor, 0.999))
        logger.info("")

    abs_v_diff = stats.v_diff__V.abs()
    logger.info("|v_diff__V|:")
    logger.info("  max: %.6g", float(abs_v_diff.max().item()))
    logger.info("  p99: %.6g", _percentile(abs_v_diff, 0.99))
    logger.info("  p99.9: %.6g", _percentile(abs_v_diff, 0.999))
    logger.info("")

    logger.info("ADC input range candidates:")
    for c in stats.candidates:
        logger.info(
            "  %s (clip_rate ~ %.4f%%, target %.4f%%):",
            c.label,
            c.observed_clip_rate * 100.0,
            c.nominal_clip_rate * 100.0,
        )
        logger.info("    input_range__V: [%.6g, %.6g]", -c.a__V, c.a__V)


_PLOT_BITS_MIN = 1
_PLOT_BITS_MAX = 12


def plot_statistics(
    stats: Statistics,
    output_dir: Path,
    *,
    bits: int,
    bins_per_code: int,
) -> None:
    """Render ``overview.png`` and per-candidate spotlight PNGs.

    Histogram bin widths are derived from the ADC code grid so the bins
    never aliases coarser than the ADC's LSB:

    * ``overview.png`` uses bin_width = LSB(max_abs candidate) / bins_per_code.
      Sub-code resolution makes the full-range hist smooth while still
      anchored to a physical reference.
    * ``spotlight_*.png`` uses bin_width = LSB(this candidate) so each bin
      represents exactly one ADC code. Bin edges line up with the code
      boundaries drawn on the same plot.

    Both views also extend the x-axis by 10 % beyond ``±A`` (same bin
    width) so samples that would be clipped are visible.

    Args:
        stats: Result of :func:`collect_statistics`.
        output_dir: Directory to write PNGs into; created if missing.
        bits: ADC bit width for the code-grid overlay.
        bins_per_code: Overview hist oversample factor; ``1`` uses one
            bin per ADC code, larger reveals sub-code structure. Spotlight
            plots always use ``1`` bin per code.

    Raises:
        RuntimeError: When matplotlib is not installed.
        ValueError: When ``bits`` is outside ``[1, 12]`` or
            ``bins_per_code`` < 1.
    """
    if not (_PLOT_BITS_MIN <= bits <= _PLOT_BITS_MAX):
        raise ValueError(f"bits = {bits} invalid: must be in [{_PLOT_BITS_MIN}, {_PLOT_BITS_MAX}]")
    if bins_per_code < 1:
        raise ValueError(f"bins_per_code = {bins_per_code} invalid: must be >= 1")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot-dir but is not installed") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    v_diff = stats.v_diff__V.cpu().numpy()
    abs_diff = stats.v_diff__V.abs().cpu().numpy()
    colors = plt.get_cmap("tab10")
    n_codes = 1 << bits

    # ----- overview.png: hist + CCDF master view -----
    # Anchor the overview bin width to the LSB of the widest candidate
    # (max_abs), then sub-divide by bins_per_code for smooth detail. This
    # guarantees each ADC code spans at least ``bins_per_code`` hist bins.
    max_abs_a = max(c.a__V for c in stats.candidates)
    overview_lsb = 2.0 * max_abs_a / n_codes
    overview_bin_width = overview_lsb / bins_per_code
    overview_xlim = 1.1 * max_abs_a
    overview_n_bins = max(2, math.ceil(2.0 * overview_xlim / overview_bin_width))
    overview_edges = np.linspace(-overview_xlim, overview_xlim, overview_n_bins + 1)
    fig, (ax_hist, ax_ccdf) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    ax_hist.hist(v_diff, bins=overview_edges, log=True, alpha=0.7)
    for i, c in enumerate(stats.candidates):
        color = colors(i % 10)
        ax_hist.axvline(c.a__V, color=color, linestyle="--", alpha=0.8, label=c.label)
        ax_hist.axvline(-c.a__V, color=color, linestyle="--", alpha=0.8)
    ax_hist.set_xlabel("v_diff [V]")
    ax_hist.set_ylabel("count (log)")
    ax_hist.set_title(
        f"ADC differential input distribution (bin = LSB_max_abs / {bins_per_code} = {overview_bin_width * 1e3:.3g} mV)"
    )
    ax_hist.legend(fontsize=8, loc="upper right")

    sorted_abs = sorted(abs_diff)
    n_total = len(sorted_abs)
    ccdf = [1.0 - (i + 1) / n_total for i in range(n_total)]
    ax_ccdf.semilogy(sorted_abs, ccdf)
    for i, c in enumerate(stats.candidates):
        color = colors(i % 10)
        ax_ccdf.axvline(c.a__V, color=color, linestyle="--", alpha=0.8, label=c.label)
    ax_ccdf.set_xlabel("|v_diff| [V]")
    ax_ccdf.set_ylabel("clip rate (log)")
    ax_ccdf.set_title("Clip rate vs. range A")
    ax_ccdf.legend(fontsize=8, loc="upper right")
    ax_ccdf.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----- spotlight_*.png: one per candidate, code grid overlaid -----
    spotlight_width = max(12.0, 6.0 + 1.2 * bits)

    for i, c in enumerate(stats.candidates):
        color = colors(i % 10)
        a = c.a__V
        lsb = 2.0 * a / n_codes

        # Bin width = LSB so every bin spans one ADC code. Extend by ~10 %
        # on each side at the same width so clipped samples remain visible.
        n_ext = max(1, math.ceil(0.1 * a / lsb))
        x_lo = -a - n_ext * lsb
        x_hi = +a + n_ext * lsb
        spotlight_edges = np.linspace(x_lo, x_hi, n_codes + 2 * n_ext + 1)

        fig, ax = plt.subplots(figsize=(spotlight_width, 4.0))
        ax.hist(v_diff, bins=spotlight_edges.tolist(), log=True, alpha=0.6, color="C0")

        # Interior code-boundary lines (light, same candidate color).
        # Drawn first so the bold +/-A lines layer on top.
        for k in range(1, n_codes):
            ax.axvline(-a + k * lsb, color=color, linewidth=0.5, alpha=0.4)

        # Bold +/-A boundary lines.
        ax.axvline(a, color=color, linewidth=1.5, alpha=1.0)
        ax.axvline(-a, color=color, linewidth=1.5, alpha=1.0)

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("v_diff [V]")
        ax.set_ylabel("count (log)")
        ax.set_title(
            f"{c.label}: A=+/-{a:.4g}V "
            f"| observed_clip={c.observed_clip_rate * 100.0:.4f}% "
            f"| bits={bits} "
            f"| n_codes={n_codes} "
            f"| LSB={lsb:.4g}V"
        )

        fig.tight_layout()
        fig.savefig(output_dir / c.filename, dpi=120, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        description="Probe the ADC analog-input distribution of an xbar and recommend input-range candidates.",
    )
    add_standard_args(parser, plot_dir=True)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg = load_tool_config(XbarAdcStatisticConfig, args.config)
    logger.info("loaded config from %s", args.config)

    distribution_path = resolve_relative_path(cfg.workload.distribution, args.config)

    device = torch.device(args.device)
    logger.info("resolved device: %s", device)

    stats = collect_statistics(
        xbar_config=cfg.xbar,
        distribution_path=distribution_path,
        weight_samples=cfg.workload.weight_samples,
        input_samples_per_weight=cfg.workload.input_samples_per_weight,
        batch_size=cfg.workload.batch_size,
        max_clip_rate_exp=cfg.statistic.max_clip_rate_exp,
        seed=cfg.workload.seed,
        device=device,
    )
    log_statistics(stats)

    if args.plot_dir is not None:
        plot_statistics(stats, args.plot_dir, bits=cfg.plot.bits, bins_per_code=cfg.plot.bins_per_code)
        logger.info(
            "wrote overview + %d spotlights to %s (bits=%d, bins_per_code=%d)",
            len(stats.candidates),
            args.plot_dir,
            cfg.plot.bits,
            cfg.plot.bins_per_code,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
