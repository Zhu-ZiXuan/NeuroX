"""Pure fitting logic for macro calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RescaleFit:
    rescale_factor: float
    sample_num: int
    r2: float
    rmse: float
    max_abs_residual: float


def fit_rescale_through_origin(code: Tensor, ideal: Tensor) -> RescaleFit:
    """Fit `ideal ~= rescale_factor * code` through the origin."""
    c = code.detach().flatten().to(torch.float64)
    y = ideal.detach().flatten().to(torch.float64)
    if c.numel() != y.numel():
        raise ValueError(f"require: code numel ({c.numel()}) == ideal numel ({y.numel()})")
    if c.numel() == 0:
        raise ValueError("require: at least one (code, ideal) sample")
    denom = float((c * c).sum())
    if denom <= 0.0:
        raise ValueError("require: code samples not all zero — a zero design matrix has no slope")
    factor = float((c * y).sum()) / denom
    residual = y - factor * c
    ss_res = float(residual.square().sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot <= 0.0 and ss_res <= 0.0 else (1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)
    return RescaleFit(
        rescale_factor=factor,
        sample_num=int(c.numel()),
        r2=r2,
        rmse=float(torch.sqrt(residual.square().mean())),
        max_abs_residual=float(residual.abs().max()),
    )


@dataclass(frozen=True)
class ValueCluster:
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
    """Cluster sorted scalar ranges by relative adjacent gaps."""
    if len(values) == 0:
        raise ValueError("require: at least one value to cluster")
    if rel_tol <= 0.0:
        raise ValueError(f"require: rel_tol ({rel_tol}) > 0")
    if max_cluster_num < 1:
        raise ValueError(f"require: max_cluster_num ({max_cluster_num}) >= 1")

    order = sorted(range(len(values)), key=lambda index: values[index])
    scale = max(abs(value) for value in values)
    gap_budget = rel_tol * scale if scale > 0.0 else 0.0
    groups: list[list[int]] = [[order[0]]]
    for index in order[1:]:
        if values[index] - values[groups[-1][-1]] > gap_budget:
            groups.append([index])
        else:
            groups[-1].append(index)

    while len(groups) > max_cluster_num:
        gaps = [values[groups[index + 1][0]] - values[groups[index][-1]] for index in range(len(groups) - 1)]
        merge_at = min(range(len(gaps)), key=lambda index: gaps[index])
        groups[merge_at : merge_at + 2] = [groups[merge_at] + groups[merge_at + 1]]

    return tuple(
        ValueCluster(
            lo=values[group[0]],
            hi=values[group[-1]],
            representative=values[group[-1]],
            member_idx=tuple(group),
        )
        for group in groups
    )
