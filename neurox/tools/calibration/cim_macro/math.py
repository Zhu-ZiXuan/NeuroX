"""Pure fitting logic for CIM-macro calibration."""

from __future__ import annotations

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
    """Fit one zero-intercept scale from paired code and ideal observations.

    Inputs are detached, flattened independently, and converted to float64 on
    their current devices. Supply finite, elementwise-paired samples with equal
    counts on the same device. Shapes may differ if flattening preserves the
    intended pairing. Python scalar statistics synchronize accelerator inputs;
    use this routine for offline fitting rather than inside compiled execution.

    Args:
        code: Finite observed code samples on the same device as the ideal
            values.
        ideal: Paired ideal samples with the same flattened element count.

    Returns:
        Fitted scale and residual statistics. Constant ideal targets have R2 of
        one only for a zero residual, otherwise zero. No positivity constraint
        is imposed on the fitted scale.

    Raises:
        ValueError: Counts differ, the sample set is empty, or all codes are
            zero.
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
    rescale_factor = float((c * y).sum()) / denom
    residual = y - rescale_factor * c
    ss_res = float(residual.square().sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot <= 0.0 and ss_res <= 0.0 else (1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0)
    return RescaleFit(
        rescale_factor=rescale_factor,
        sample_num=int(c.numel()),
        r2=r2,
        rmse=float(torch.sqrt(residual.square().mean())),
        max_abs_residual=float(residual.abs().max()),
    )
