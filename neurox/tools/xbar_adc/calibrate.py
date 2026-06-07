"""CLI: fit ``rescale_factor`` mapping physical ADC codes back to ideal VMM.

See also:
    docs/dev/modules/tools/xbar_adc/calibrate.md
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint, McsSarAdcConfig, SarAdcMonoConfig
from neurox.tools.logging import config_tool_logging
from neurox.tools.xbar_adc._sampling import (
    build_offset_1t1r_xbar_all_off,
    load_distribution,
    make_generator,
    resolve_device,
    sample_w,
    sample_x_batches,
)

logger = logging.getLogger(__name__)

_LOSSLESS_OP = AdcOperationPoint(adc_mode=0, adc_bits=0)


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted ``rescale_factor`` + diagnostics for one ``adc_mode``.

    Attributes:
        adc_mode: Operating-point index calibrated.
        max_bits: ``xbar.adc_max_bits``; bit width of the LS fit.
        r_max: ``rescale_factor`` at ``max_bits``.
        derived_rescale: ``{b: r_max * 2**(max_bits - b)}`` for
            ``b ∈ [1, max_bits]``.
        total_pairs / valid_pairs / saturated_pairs_excluded /
            saturation_rate: Sample counting / exclusion stats.
        mse / rmse / mae / residual_mean / residual_std / max_abs_residual:
            Residual diagnostics on valid samples.
        phys_codes / ideal_vmm: Concatenated CPU float64 1-D pairs.
        sat_mask: Boolean mask (same shape as ``phys_codes``); True where
            ``phys_codes`` was saturated and excluded from the LS fit.
        supports_flexible_bits: True for SAR-family ADCs that can run at
            ``bits < max_bits`` with the same V_ref / range. Drives whether
            the derived rescale table is emitted in :func:`log_calibration`.
        adc_instance_count: Number of physically-identical ``bl_adc``
            instances in the xbar; ``(phys_code, ideal_vmm)`` pairs are
            pooled across all of them by design (identical circuits
            sharing the same bias).
        distribution_source: ``"uniform"`` or the distribution-TOML path.
    """

    adc_mode: int
    max_bits: int
    r_max: float
    derived_rescale: dict[int, float]
    total_pairs: int
    valid_pairs: int
    saturated_pairs_excluded: int
    saturation_rate: float
    mse: float
    rmse: float
    mae: float
    residual_mean: float
    residual_std: float
    max_abs_residual: float
    phys_codes: Tensor
    ideal_vmm: Tensor
    sat_mask: Tensor
    supports_flexible_bits: bool
    adc_instance_count: int
    distribution_source: str


# ---------------------------------------------------------------------------
# LS fit
# ---------------------------------------------------------------------------


def fit_rescale_factor(phys_codes: Tensor, ideal_vmm: Tensor) -> float:
    """Solve ``min_r Σ (p_i · r - y_i)^2`` in float64.

    Args:
        phys_codes: Physical ADC codes ``p_i``.
        ideal_vmm: Ideal integer VMM targets ``y_i``.

    Returns:
        Closed-form solution ``r = Σ p_i y_i / Σ p_i²``.

    Raises:
        ValueError: When ``Σ p_i² == 0`` (all valid codes are zero).
    """
    p = phys_codes.to(torch.float64)
    y = ideal_vmm.to(torch.float64)
    numerator = float((p * y).sum().item())
    denominator = float((p * p).sum().item())
    if denominator == 0.0:
        raise ValueError(
            "cannot calibrate rescale_factor: all valid physical codes are zero "
            "(check adc_mode / range; verify ADC config matches the workload)"
        )
    return numerator / denominator


def derive_rescale_for_bits(r_max: float, max_bits: int) -> dict[int, float]:
    """``rescale(b) = r_max * 2 ** (max_bits - b)`` for ``b ∈ [1, max_bits]``."""
    if max_bits < 1:
        raise ValueError(f"max_bits ({max_bits}) must be >= 1")
    return {b: r_max * (2 ** (max_bits - b)) for b in range(1, max_bits + 1)}


def saturation_mask(phys_codes: Tensor, max_bits: int) -> Tensor:
    """Boolean mask — True where ``phys_codes`` is at a signed endpoint.

    ADC.convert returns signed codes in ``[-2**(b-1), 2**(b-1) - 1]``; the
    endpoints carry no linear-region information and must always be excluded
    from the LS fit.
    """
    lower = -(1 << (max_bits - 1))
    upper = (1 << (max_bits - 1)) - 1
    return (phys_codes == lower) | (phys_codes == upper)


def supports_flexible_bits(adc_config: object) -> bool:
    """True for ADC families whose bit width is freely selectable within one mode.

    SAR-family ADCs can run at any ``bits <= max_bits`` with the same V_ref,
    so their lower-bit ``rescale_factor`` follows the power-of-2 derivation
    rule. Other ADCs (e.g. ``GeneralADC``) are bit-width-fixed; the derived
    table is meaningless for them.
    """
    return isinstance(adc_config, (SarAdcMonoConfig, McsSarAdcConfig))


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------


def collect_calibration(
    *,
    config_path: Path,
    distribution_path: Path | None,
    adc_mode: int,
    weight_samples: int,
    input_samples_per_weight: int,
    batch_size: int,
    seed: int | None,
    device: torch.device,
) -> CalibrationResult:
    """Run the dual-xbar sweep and fit the mode's ``rescale_factor``.

    Saturated samples (codes at the signed extremes) are always excluded
    from the LS fit — they carry no linear-region information.

    Args:
        config_path: chip xbar TOML.
        distribution_path: distribution TOML; ``None`` means uniform.
        adc_mode: Operating-point index to calibrate.
        weight_samples / input_samples_per_weight / batch_size: see
            :func:`statistic.collect_statistics`.
        seed: optional generator seed.
        device: resolved torch device.

    Raises:
        ValueError: When ``adc_mode`` is out of range, sweep params are
            non-positive, all valid codes are zero, or the fitted
            ``r_max <= 0`` (broken sign convention or grossly misconfigured ADC).
    """
    if weight_samples <= 0:
        raise ValueError(f"weight_samples ({weight_samples}) must be > 0")
    if input_samples_per_weight <= 0:
        raise ValueError(f"input_samples_per_weight ({input_samples_per_weight}) must be > 0")
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > 0")

    physical = build_offset_1t1r_xbar_all_off(config_path, device=device)
    ideal = physical.to_ideal().to(device)
    ideal.eval()
    ideal.fabricate()

    if not (0 <= adc_mode < physical.adc_mode_num):
        raise ValueError(f"adc_mode ({adc_mode}) must lie in [0, {physical.adc_mode_num})")

    distribution = load_distribution(distribution_path, physical)
    generator = make_generator(seed, device)
    max_bits = physical.adc_max_bits
    phys_op = AdcOperationPoint(adc_mode=adc_mode, adc_bits=max_bits)
    adc_instance_count = int(math.prod(physical.readout.bl_adc._inst_shape))
    flexible = supports_flexible_bits(physical.readout.config.adc_config)

    logger.info("xbar built on device=%s; distribution_source=%s", device, distribution.source)
    logger.info(
        "calibration: adc_mode=%d, max_bits=%d, weight_samples=%d, input_samples_per_weight=%d",
        adc_mode,
        max_bits,
        weight_samples,
        input_samples_per_weight,
    )

    phys_buf: list[Tensor] = []
    ideal_buf: list[Tensor] = []
    for i, w in enumerate(sample_w(distribution, physical, n=weight_samples, device=device, generator=generator)):
        physical.program(w)
        ideal.program(w)
        for x in sample_x_batches(
            distribution,
            physical,
            n_total=input_samples_per_weight,
            batch_size=batch_size,
            device=device,
            generator=generator,
        ):
            phys_code = physical.vec_mat_mul(x, adc_operation_point=phys_op)
            ideal_vmm = ideal.vec_mat_mul(x, adc_operation_point=_LOSSLESS_OP)
            phys_buf.append(phys_code.detach().cpu().flatten().to(torch.float64))
            ideal_buf.append(ideal_vmm.detach().cpu().flatten().to(torch.float64))
        logger.info("weight sample %d/%d done", i + 1, weight_samples)

    phys_codes = torch.cat(phys_buf)
    ideal_vmm = torch.cat(ideal_buf)
    total_pairs = int(phys_codes.numel())

    sat_mask = saturation_mask(phys_codes, max_bits)
    valid_mask = ~sat_mask

    fit_phys = phys_codes[valid_mask]
    fit_ideal = ideal_vmm[valid_mask]
    valid_pairs = int(fit_phys.numel())
    saturated_excluded = int(sat_mask.sum().item())
    saturation_rate = float(sat_mask.to(torch.float64).mean().item())

    if valid_pairs == 0:
        raise ValueError(
            f"cannot calibrate: zero valid samples after saturation filter "
            f"(saturation_rate={saturation_rate:.6f}); "
            "ADC range is too tight for this workload — revisit V_ref with statistic_xbar_adc."
        )

    r_max = fit_rescale_factor(fit_phys, fit_ideal)
    if r_max <= 0.0:
        raise ValueError(
            f"fitted rescale_factor is non-positive ({r_max:.6g}). "
            "rescale_factor is a strictly positive scalar by architectural invariant. "
            "Possible causes: (a) ADC.convert is not returning signed-centred codes "
            "(check sign convention), (b) ADC config grossly misconfigured for the workload, "
            "(c) per-mode distribution mismatched with the V_ref of this mode."
        )

    derived = derive_rescale_for_bits(r_max, max_bits)

    residual = fit_phys * r_max - fit_ideal
    abs_res = residual.abs()
    return CalibrationResult(
        adc_mode=adc_mode,
        max_bits=max_bits,
        r_max=r_max,
        derived_rescale=derived,
        total_pairs=total_pairs,
        valid_pairs=valid_pairs,
        saturated_pairs_excluded=saturated_excluded,
        saturation_rate=saturation_rate,
        mse=float((residual * residual).mean().item()),
        rmse=float((residual * residual).mean().sqrt().item()),
        mae=float(abs_res.mean().item()),
        residual_mean=float(residual.mean().item()),
        residual_std=float(residual.std().item()),
        max_abs_residual=float(abs_res.max().item()),
        phys_codes=phys_codes,
        ideal_vmm=ideal_vmm,
        sat_mask=sat_mask,
        supports_flexible_bits=flexible,
        adc_instance_count=adc_instance_count,
        distribution_source=distribution.source,
    )


# ---------------------------------------------------------------------------
# Logger output + plot
# ---------------------------------------------------------------------------


def log_calibration(result: CalibrationResult) -> None:
    """Emit the structured per-section summary + TOML snippet."""
    logger.info("ADC rescale calibration")
    logger.info("  sample_source: synthetic")
    logger.info("  distribution: %s", result.distribution_source)
    logger.info("  adc_mode: %d", result.adc_mode)
    logger.info("  calibration_bits: %d", result.max_bits)
    logger.info("  total_pairs: %d", result.total_pairs)
    logger.info("  valid_pairs: %d", result.valid_pairs)
    logger.info("  saturated_pairs_excluded: %d", result.saturated_pairs_excluded)
    logger.info("  saturation_rate: %.6f", result.saturation_rate)
    logger.info(
        "  adc_instance_count: %d (pooled across all instances; identical circuits + shared bias)",
        result.adc_instance_count,
    )
    logger.info("")
    logger.info("rescale_factor_at_max_bits: %.6g", result.r_max)
    logger.info("rmse: %.6g", result.rmse)
    logger.info("mae: %.6g", result.mae)
    logger.info("residual_mean: %.6g", result.residual_mean)
    logger.info("residual_std: %.6g", result.residual_std)
    logger.info("max_abs_residual: %.6g", result.max_abs_residual)
    logger.info("")

    if result.supports_flexible_bits:
        logger.info(
            "Derived rescale factors (informational; NOT calibrated, follow "
            "R(b) = R_max * 2 ** (max_bits - b) — copy into xbar config manually if needed):"
        )
        for b in sorted(result.derived_rescale.keys(), reverse=True):
            marker = "  [<- calibrated]" if b == result.max_bits else ""
            logger.info("  bits=%d: %.6g%s", b, result.derived_rescale[b], marker)
        logger.info("")

    logger.info("Copyable config snippet (paste under xbar.adc_calibration; calibrated entry only):")
    logger.info("[[adc_calibration]]")
    logger.info("adc_mode = %d", result.adc_mode)
    logger.info("adc_bits = %d", result.max_bits)
    logger.info("rescale_factor = %.10g", result.r_max)
    logger.info("")


def plot_calibration(result: CalibrationResult, output_path: Path) -> None:
    """Render a two-panel PNG: fit scatter + residual histogram.

    Raises:
        RuntimeError: when matplotlib is not installed.
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot but is not installed") from exc

    valid_mask = ~result.sat_mask
    fit_phys = result.phys_codes[valid_mask].cpu().numpy()
    fit_ideal = result.ideal_vmm[valid_mask].cpu().numpy()
    sat_phys = result.phys_codes[result.sat_mask].cpu().numpy()
    sat_ideal = result.ideal_vmm[result.sat_mask].cpu().numpy()

    fig, (ax_scatter, ax_resid) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: fit scatter with the calibrated line and saturated highlight.
    ax_scatter.scatter(fit_phys, fit_ideal, s=2, alpha=0.4, label="valid")
    if sat_phys.size > 0:
        ax_scatter.scatter(sat_phys, sat_ideal, s=4, alpha=0.6, color="red", label="saturated")
    x_lo = float(min(fit_phys.min(), sat_phys.min() if sat_phys.size else fit_phys.min()))
    x_hi = float(max(fit_phys.max(), sat_phys.max() if sat_phys.size else fit_phys.max()))
    xs = [x_lo, x_hi]
    ys = [x * result.r_max for x in xs]
    ax_scatter.plot(xs, ys, color="black", linewidth=1.0, label=f"r_max={result.r_max:.4g}")
    ax_scatter.set_xlabel("phys_code")
    ax_scatter.set_ylabel("ideal VMM")
    ax_scatter.set_title(f"Calibration fit @ adc_mode={result.adc_mode}, bits={result.max_bits}")
    ax_scatter.legend(fontsize=8, loc="best")
    ax_scatter.grid(True, alpha=0.3)

    # Right: residual histogram with mean / mean±std markers.
    residual = fit_phys * result.r_max - fit_ideal
    ax_resid.hist(residual, bins=128, alpha=0.7)
    ax_resid.axvline(result.residual_mean, color="black", linestyle="-", label=f"mean={result.residual_mean:.3g}")
    ax_resid.axvline(
        result.residual_mean + result.residual_std,
        color="black",
        linestyle="--",
        label=f"+/- std ({result.residual_std:.3g})",
    )
    ax_resid.axvline(result.residual_mean - result.residual_std, color="black", linestyle="--")
    ax_resid.set_xlabel("residual = phys_code * r_max - ideal_vmm")
    ax_resid.set_ylabel("count")
    ax_resid.set_title("Residuals")
    ax_resid.legend(fontsize=8, loc="upper right")
    ax_resid.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the rescale_factor that maps physical ADC codes back to ideal VMM outputs.",
    )
    parser.add_argument("--xbar-config", type=Path, required=True, help="Chip xbar TOML path")
    parser.add_argument("--adc-mode", type=int, required=True, help="Operating-point index to calibrate")
    parser.add_argument(
        "--distribution",
        type=Path,
        default=None,
        help="Optional synthetic-workload distribution TOML; omitted -> uniform",
    )
    parser.add_argument("--weight-samples", type=int, default=32, help="Independent programmed-state samples")
    parser.add_argument(
        "--input-samples-per-weight",
        type=int,
        default=1024,
        help="Per-weight primitive input-vector count",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Per-VMM input batch cap")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device; "auto" picks cuda if available else cpu',
    )
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG output path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logger level (e.g. DEBUG, INFO)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_tool_logging(level=getattr(logging, args.log_level.upper()))

    device = resolve_device(args.device)
    logger.info("resolved device: %s", device)

    result = collect_calibration(
        config_path=args.xbar_config,
        distribution_path=args.distribution,
        adc_mode=args.adc_mode,
        weight_samples=args.weight_samples,
        input_samples_per_weight=args.input_samples_per_weight,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    log_calibration(result)

    if args.plot is not None:
        plot_calibration(result, args.plot)
        logger.info("wrote plot to %s", args.plot)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
