"""Pure math for the generic macro-ADC calibration tools.

Every function here is deterministic tensor / scalar math with no module
building, no file IO, and no plotting — the unit-testable core the three CLI
entries `rescale_fit`, `threshold_probe`, and `mode_derive` consume.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common import TensorDataClassBase

# ---------------------------------------------------------------------------
# Zero-through-origin rescale fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RescaleFit:
    """One-parameter least-squares fit `ideal ~= rescale_factor * code`."""

    rescale_factor: float
    """Fitted slope through the origin."""
    sample_num: int
    """Samples entering the fit, after exclusion."""
    r2: float
    """Coefficient of determination of the through-origin model against the
    mean-of-`ideal` baseline; 1.0 when `ideal` is constant and the residuals
    are zero."""
    rmse: float
    """Root-mean-square residual `ideal - rescale_factor * code`."""
    max_abs_residual: float
    """Largest absolute residual."""


def fit_rescale_through_origin(code: Tensor, ideal: Tensor) -> RescaleFit:
    """LS-fit the zero-through-origin rescale `ideal ~= r * code`.

    Args:
        code: ADC code samples, flattened internally. Must not be all zero —
            a zero design matrix has no slope.
            Shape: `[...]`.
        ideal: Ideal-value samples, same element count as `code`.
            Shape: `[...]`.

    Returns:
        The fit, accumulated in float64.

    Raises:
        ValueError: On element-count mismatch, an empty input, or an all-zero
            `code`.
    """
    c = code.detach().flatten().to(torch.float64)
    y = ideal.detach().flatten().to(torch.float64)
    if c.numel() != y.numel():
        raise ValueError(f"require: code numel ({c.numel()}) == ideal numel ({y.numel()})")
    if c.numel() == 0:
        raise ValueError("require: at least one (code, ideal) sample")
    denom = float((c * c).sum())
    if denom <= 0.0:
        raise ValueError("require: code samples not all zero — a zero design matrix has no slope")
    r = float((c * y).sum()) / denom
    residual = y - r * c
    ss_res = float((residual * residual).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot <= 0.0 and ss_res <= 0.0 else (1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)
    return RescaleFit(
        rescale_factor=r,
        sample_num=int(c.numel()),
        r2=r2,
        rmse=float(torch.sqrt(residual.square().mean())),
        max_abs_residual=float(residual.abs().max()),
    )


# ---------------------------------------------------------------------------
# Input-code band statistics + threshold placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputCodeBand:
    """Observed analog band at one integer ADC input code."""

    input_code: int
    """The mapped quantization input the converter discriminates on."""
    lo: float
    """Minimum analog input observed at this code."""
    hi: float
    """Maximum analog input observed at this code."""
    mean: float
    """Mean analog input observed at this code."""
    count: int
    """Samples in the band."""


def band_stats(
    input_code: Tensor,
    analog: Tensor,
    *,
    adc_input_code_range: tuple[int, int],
) -> tuple[InputCodeBand, ...]:
    """Group analog samples by integer input code into per-code bands.

    A pair whose input code falls outside `adc_input_code_range` is masked
    out of both streams: the converter resolves no tap there, so its analog
    value bounds no band of this grid.

    Args:
        input_code: Integer ADC input code per sample, flattened internally.
            Shape: `[...]`.
        analog: Analog input per sample, same element count.
            Shape: `[...]`.
        adc_input_code_range: Inclusive grid bounds `(lower, upper)`.

    Returns:
        One band per code in the range, ascending.

    Raises:
        ValueError: On element-count mismatch, a degenerate or negative
            range, or an in-range code with no samples (coverage gap).
    """
    m = input_code.detach().flatten().to(torch.int64)
    a = analog.detach().flatten().to(torch.float64)
    if m.numel() != a.numel():
        raise ValueError(f"require: input_code numel ({m.numel()}) == analog numel ({a.numel()})")
    lower, upper = adc_input_code_range
    if lower < 0:
        raise ValueError(f"require: adc_input_code_range lower ({lower}) >= 0")
    if upper <= lower:
        raise ValueError(f"require: adc_input_code_range upper ({upper}) > lower ({lower})")
    keep = (m >= lower) & (m <= upper)
    m = m[keep]
    a = a[keep]
    bands: list[InputCodeBand] = []
    missing: list[int] = []
    for k in range(lower, upper + 1):
        sel = a[m == k]
        if sel.numel() == 0:
            missing.append(k)
            continue
        bands.append(
            InputCodeBand(
                input_code=k,
                lo=float(sel.min()),
                hi=float(sel.max()),
                mean=float(sel.mean()),
                count=int(sel.numel()),
            )
        )
    if missing:
        raise ValueError(f"band coverage gap: no samples observed at input code {missing} (grid {lower} .. {upper})")
    return tuple(bands)


@dataclass(frozen=True)
class ThresholdPlacement:
    """Mid-point threshold ladder + band-margin diagnostics for one mode."""

    thresholds: tuple[float, ...]
    """One code-boundary threshold per adjacent band pair,
    `t[k] = (hi(k) + lo(k+1)) / 2`."""
    margins: tuple[float, ...]
    """Per-boundary band separation `lo(k+1) - hi(k)`; a negative entry means
    the adjacent bands overlap."""
    min_margin: float
    """Smallest margin — the report headline."""
    min_margin_boundary: int
    """The `k` of the smallest margin, at the `k`/`k+1` boundary."""
    monotone: bool
    """True iff the band means are strictly increasing and so is the
    threshold ladder."""


def place_thresholds(bands: Sequence[InputCodeBand]) -> ThresholdPlacement:
    """Place mid-point thresholds between adjacent input-code bands.

    Overlap and non-monotonicity never raise — the placement carries the
    `margins` / `monotone` diagnostics so the caller can report a failed band
    separation instead of crashing mid-report.

    Args:
        bands: Ascending complete band grid, at least two bands.

    Raises:
        ValueError: Fewer than two bands.
    """
    if len(bands) < 2:
        raise ValueError(f"require: at least 2 bands to place a threshold; got {len(bands)}")
    thresholds: list[float] = []
    margins: list[float] = []
    for lower, upper in itertools.pairwise(bands):
        thresholds.append(0.5 * (lower.hi + upper.lo))
        margins.append(upper.lo - lower.hi)
    means = [b.mean for b in bands]
    monotone = all(b > a for a, b in itertools.pairwise(means)) and all(
        b > a for a, b in itertools.pairwise(thresholds)
    )
    min_idx = min(range(len(margins)), key=lambda i: margins[i])
    return ThresholdPlacement(
        thresholds=tuple(thresholds),
        margins=tuple(margins),
        min_margin=margins[min_idx],
        min_margin_boundary=min_idx,
        monotone=monotone,
    )


@dataclass(frozen=True)
class LinearFit:
    """Ordinary least-squares line `y ~= slope * x + intercept`."""

    slope: float
    intercept: float
    r2: float


def fit_linear(x: Tensor, y: Tensor) -> LinearFit:
    """OLS line fit in float64 — the grid-curve `I(M)` diagnostic.

    Args:
        x: Abscissa samples, flattened internally; at least 2 distinct.
            Shape: `[...]`.
        y: Ordinate samples, same element count.
            Shape: `[...]`.

    Raises:
        ValueError: On element-count mismatch, fewer than 2 samples, or a
            degenerate abscissa.
    """
    xf = x.detach().flatten().to(torch.float64)
    yf = y.detach().flatten().to(torch.float64)
    if xf.numel() != yf.numel():
        raise ValueError(f"require: x numel ({xf.numel()}) == y numel ({yf.numel()})")
    if xf.numel() < 2:
        raise ValueError("require: at least 2 samples for a line fit")
    x_mean = xf.mean()
    y_mean = yf.mean()
    var_x = float(((xf - x_mean) ** 2).sum())
    if var_x <= 0.0:
        raise ValueError("require: at least 2 distinct x values for a line fit")
    cov = float(((xf - x_mean) * (yf - y_mean)).sum())
    slope = cov / var_x
    intercept = float(y_mean) - slope * float(x_mean)
    residual = yf - (slope * xf + intercept)
    ss_res = float((residual * residual).sum())
    ss_tot = float(((yf - y_mean) ** 2).sum())
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return LinearFit(slope=slope, intercept=intercept, r2=r2)


# ---------------------------------------------------------------------------
# 1-D value clustering (mode derivation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueCluster:
    """One cluster of scalar range values."""

    lo: float
    """Smallest member value."""
    hi: float
    """Largest member value."""
    representative: float
    """The largest member, so a mode sized from it covers every member's
    range."""
    member_idx: tuple[int, ...]
    """Member indices into the input sequence, ascending by value."""


def cluster_values(
    values: Sequence[float],
    *,
    rel_tol: float,
    max_cluster_num: int,
) -> tuple[ValueCluster, ...]:
    """Cluster scalars into ascending contiguous groups by relative gap.

    Deterministic 1-D agglomerative clustering: sort, split wherever the gap
    between consecutive values exceeds `rel_tol * max|value|`, then while more
    than `max_cluster_num` clusters remain merge the adjacent pair with the
    smallest inter-cluster gap.

    Args:
        values: Scalar values, e.g. learned per-layer range params.
        rel_tol: Gap threshold relative to the largest absolute value.
        max_cluster_num: Hard cap on the cluster count.

    Returns:
        Ascending clusters covering every input index exactly once.

    Raises:
        ValueError: On empty input or non-positive `rel_tol` /
            `max_cluster_num`.
    """
    if len(values) == 0:
        raise ValueError("require: at least one value to cluster")
    if rel_tol <= 0.0:
        raise ValueError(f"require: rel_tol ({rel_tol}) > 0")
    if max_cluster_num < 1:
        raise ValueError(f"require: max_cluster_num ({max_cluster_num}) >= 1")

    order = sorted(range(len(values)), key=lambda i: values[i])
    scale = max(abs(v) for v in values)
    gap_budget = rel_tol * scale if scale > 0.0 else 0.0

    groups: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        prev_val = values[groups[-1][-1]]
        if values[idx] - prev_val > gap_budget:
            groups.append([idx])
        else:
            groups[-1].append(idx)

    while len(groups) > max_cluster_num:
        # Merge the adjacent pair with the smallest inter-cluster gap.
        gaps = [values[groups[i + 1][0]] - values[groups[i][-1]] for i in range(len(groups) - 1)]
        j = min(range(len(gaps)), key=lambda i: gaps[i])
        groups[j : j + 2] = [groups[j] + groups[j + 1]]

    return tuple(
        ValueCluster(
            lo=values[g[0]],
            hi=values[g[-1]],
            representative=values[g[-1]],
            member_idx=tuple(g),
        )
        for g in groups
    )


# ---------------------------------------------------------------------------
# Calibration-pair sample filter (rescale fit)
# ---------------------------------------------------------------------------


class FitSampleFilter(TensorDataClassBase):
    """Keep mask + per-cause drop counts for one mode's calibration pairs.

    `sample_num` is the flattened input pairs' common element count.
    """

    keep: Tensor
    """Boolean mask over the flattened input pairs.
    Shape: `[sample_num]`."""
    range_dropped_num: int
    """Pairs whose ADC input code falls outside the mode's design domain on
    the ideal axis."""
    saturated_num: int
    """Top-code-saturated pairs, carrying no linear-region information; the
    two drop causes may overlap."""


def filter_fit_samples(
    code: Tensor,
    adc_input_code: Tensor,
    *,
    adc_input_code_range: tuple[int, int],
    top_code: int,
) -> FitSampleFilter:
    """Select the calibration pairs entering a mode's rescale fit.

    A pair survives iff its ADC input code lies inside the inclusive
    `adc_input_code_range` and `code < top_code`.

    Args:
        code: ADC code per pair, flattened internally.
            Shape: `[...]`.
        adc_input_code: Ideal-side ADC input code per pair — the exact MAC
            dot mapped onto the macro's converter axis, same element count.
            Shape: `[...]`.
        adc_input_code_range: The mode's inclusive input-code domain.
        top_code: The quantizer's top code at the fitted bit width.

    Raises:
        ValueError: On element-count mismatch.
    """
    c = code.detach().flatten()
    m = adc_input_code.detach().flatten()
    if c.numel() != m.numel():
        raise ValueError(f"require: code numel ({c.numel()}) == adc_input_code numel ({m.numel()})")
    lower, upper = adc_input_code_range
    in_range = (m >= lower) & (m <= upper)
    unsaturated = c < top_code
    return FitSampleFilter(
        keep=in_range & unsaturated,
        range_dropped_num=int((~in_range).sum()),
        saturated_num=int((~unsaturated).sum()),
    )
