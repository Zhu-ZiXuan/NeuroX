"""Pure math for the generic macro-ADC calibration tools.

Every function here is deterministic, tensor/scalar math with no module
building, no file IO, and no plotting — the unit-testable core the three
CLI entries (:mod:`.rescale_fit`, :mod:`.threshold_probe`,
:mod:`.mode_derive`) consume.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Zero-through-origin rescale fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RescaleFit:
    """One-parameter least-squares fit ``ideal ~= rescale_factor * code``.

    Attributes:
        rescale_factor: Fitted slope through the origin.
        sample_num: Number of samples entering the fit (post-exclusion).
        r2: Coefficient of determination of the through-origin model
            against the mean-of-``ideal`` baseline; 1.0 when ``ideal`` is
            constant and the residuals are zero.
        rmse: Root-mean-square residual ``ideal - rescale_factor * code``.
        max_abs_residual: Largest absolute residual.
    """

    rescale_factor: float
    sample_num: int
    r2: float
    rmse: float
    max_abs_residual: float


def fit_rescale_through_origin(code: Tensor, ideal: Tensor) -> RescaleFit:
    """LS-fit the zero-through-origin rescale ``ideal ~= r * code``.

    Args:
        code: ADC code samples (any shape; flattened). Must not be all
            zero — a zero design matrix has no slope.
        ideal: Ideal-value samples, same element count as ``code``.

    Returns:
        The fitted :class:`RescaleFit` (float64 accumulation).

    Raises:
        ValueError: On element-count mismatch or an all-zero ``code``.
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
# Magnitude-band statistics + threshold placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MagnitudeBand:
    """Observed analog band at one integer magnitude.

    Attributes:
        magnitude: Integer per-phase MAC magnitude ``|M|`` of the band.
        lo: Minimum analog input observed at this magnitude.
        hi: Maximum analog input observed at this magnitude.
        mean: Mean analog input observed at this magnitude.
        count: Number of samples in the band.
    """

    magnitude: int
    lo: float
    hi: float
    mean: float
    count: int


def band_stats(magnitude: Tensor, analog: Tensor, *, m_max: int) -> tuple[MagnitudeBand, ...]:
    """Group analog samples by integer magnitude into per-``|M|`` bands.

    Magnitudes above ``m_max`` fold into the top band (the quantizer clips
    them to the top code, so their analog values legitimately bound the top
    band from above).

    Args:
        magnitude: Integer ``|M|`` per sample (any shape; flattened).
        analog: Analog input per sample, same element count.
        m_max: Top code of the grid; bands cover ``0 .. m_max``.

    Returns:
        One :class:`MagnitudeBand` per magnitude ``0 .. m_max``, ascending.

    Raises:
        ValueError: On element-count mismatch, a negative magnitude, or a
            magnitude in ``0 .. m_max`` with no samples (coverage gap).
    """
    m = magnitude.detach().flatten().to(torch.int64)
    a = analog.detach().flatten().to(torch.float64)
    if m.numel() != a.numel():
        raise ValueError(f"require: magnitude numel ({m.numel()}) == analog numel ({a.numel()})")
    if m.numel() > 0 and int(m.min()) < 0:
        raise ValueError(f"require: magnitudes non-negative; got min {int(m.min())}")
    if m_max < 1:
        raise ValueError(f"require: m_max ({m_max}) >= 1")
    m = m.clamp_max(m_max)
    bands: list[MagnitudeBand] = []
    missing: list[int] = []
    for k in range(m_max + 1):
        sel = a[m == k]
        if sel.numel() == 0:
            missing.append(k)
            continue
        bands.append(
            MagnitudeBand(
                magnitude=k,
                lo=float(sel.min()),
                hi=float(sel.max()),
                mean=float(sel.mean()),
                count=int(sel.numel()),
            )
        )
    if missing:
        raise ValueError(f"band coverage gap: no samples observed at |M| in {missing} (grid top m_max={m_max})")
    return tuple(bands)


@dataclass(frozen=True)
class ThresholdPlacement:
    """Mid-point threshold ladder + band-margin diagnostics for one mode.

    Attributes:
        thresholds: ``m_max`` code-boundary thresholds,
            ``t[k] = (hi(k) + lo(k+1)) / 2`` for ``k = 0 .. m_max - 1``.
        margins: Per-boundary band separation ``lo(k+1) - hi(k)``; a
            negative entry means the adjacent bands overlap.
        min_margin: Smallest margin (the report headline).
        min_margin_boundary: ``k`` of the smallest margin (the ``k``/``k+1``
            boundary).
        monotone: True iff the band means are strictly increasing AND the
            threshold ladder is strictly increasing.
    """

    thresholds: tuple[float, ...]
    margins: tuple[float, ...]
    min_margin: float
    min_margin_boundary: int
    monotone: bool


def place_thresholds(bands: Sequence[MagnitudeBand]) -> ThresholdPlacement:
    """Place mid-point thresholds between adjacent magnitude bands.

    Never raises on overlap or non-monotonicity — the placement carries
    the diagnostics (``margins`` / ``monotone``) so the caller can report
    a failed band separation instead of crashing mid-report.

    Args:
        bands: Ascending complete band grid from :func:`band_stats`
            (at least two bands).
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
    """Ordinary least-squares line ``y ~= slope * x + intercept``."""

    slope: float
    intercept: float
    r2: float


def fit_linear(x: Tensor, y: Tensor) -> LinearFit:
    """OLS line fit (float64) — the grid-curve ``I(M)`` diagnostic.

    Args:
        x: Abscissa samples (any shape; flattened), at least 2 distinct.
        y: Ordinate samples, same element count.
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
    """One cluster of scalar range values.

    Attributes:
        lo: Smallest member value.
        hi: Largest member value.
        representative: Cluster representative — the largest member, so a
            mode sized from it covers every member's range.
        member_idx: Indices (into the input sequence) of the members,
            ascending by value.
    """

    lo: float
    hi: float
    representative: float
    member_idx: tuple[int, ...]


def cluster_values(
    values: Sequence[float],
    *,
    rel_tol: float,
    max_cluster_num: int,
) -> tuple[ValueCluster, ...]:
    """Cluster scalars into ascending contiguous groups by relative gap.

    Deterministic 1-D agglomerative clustering: sort, split wherever the
    gap between consecutive values exceeds ``rel_tol * max|value|``, then
    while more than ``max_cluster_num`` clusters remain merge the adjacent
    pair with the smallest inter-cluster gap.

    Args:
        values: Scalar values (e.g. learned per-layer range params).
        rel_tol: Gap threshold relative to the largest absolute value.
        max_cluster_num: Hard cap on the cluster count.

    Returns:
        Ascending clusters covering every input index exactly once.

    Raises:
        ValueError: On empty input or non-positive ``rel_tol`` /
            ``max_cluster_num``.
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


@dataclass(frozen=True)
class FitSampleFilter:
    """Keep mask + per-cause drop counts for one mode's calibration pairs.

    Attributes:
        keep: Boolean mask over the flattened input pairs.
        range_dropped_num: Pairs with ``ideal_abs > range_limit`` (outside
            the mode's design domain on the ideal axis).
        saturated_num: Pairs with ``code >= top_code`` (top-code-saturated;
            no linear-region information). The two causes may overlap.
    """

    keep: Tensor
    range_dropped_num: int
    saturated_num: int


def filter_fit_samples(code: Tensor, ideal_abs: Tensor, *, range_limit: float, top_code: int) -> FitSampleFilter:
    """Select the calibration pairs entering a mode's rescale fit.

    A pair survives iff ``ideal_abs <= range_limit`` AND ``code < top_code``.

    Args:
        code: ADC code per pair (any shape; flattened).
        ideal_abs: Ideal integer magnitude per pair, same element count.
        range_limit: The mode's design range (ideal-axis domain bound).
        top_code: The quantizer's top code at the fitted bit width.

    Raises:
        ValueError: On element-count mismatch.
    """
    c = code.detach().flatten()
    m = ideal_abs.detach().flatten()
    if c.numel() != m.numel():
        raise ValueError(f"require: code numel ({c.numel()}) == ideal_abs numel ({m.numel()})")
    in_range = m.to(torch.float64) <= range_limit
    unsaturated = c < top_code
    return FitSampleFilter(
        keep=in_range & unsaturated,
        range_dropped_num=int((~in_range).sum()),
        saturated_num=int((~unsaturated).sum()),
    )
