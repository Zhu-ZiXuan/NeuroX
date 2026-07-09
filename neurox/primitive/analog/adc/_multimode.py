"""Shared utilities for multi-mode physics-based ADCs.

Each mode's uniform code boundaries sit at ``c · LSB`` (floor
semantics); the full placement is specified in the reference spec.

See also:
    docs/reference/primitive/analog/adc/base.md
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from neurox.common.quant import floor_bucketize

from .base import ADCMode


def derive_uniform_boundaries(mode: ADCMode, *, dtype: torch.dtype) -> Tensor:
    """Build the floor-style boundary list for one ADC mode.

    Boundaries are placed at code edges ``c · LSB`` so that
    ``code = bucketize(signal, boundaries)`` is the floor of
    ``signal / LSB`` (clipped to ``[0, n_codes - 1]``).

    Args:
        mode: Active ADC mode.
        dtype: Float dtype for the resulting tensor.

    Returns:
        Tensor of shape ``[n_codes - 1]`` in increasing order.
    """
    n_codes = mode.n_codes
    lsb = mode.lsb
    return torch.tensor([c * lsb for c in range(1, n_codes)], dtype=dtype)


def select_mode(modes: Sequence[ADCMode], mode: int) -> ADCMode:
    """Validate ``mode`` against ``modes`` and return the chosen one.

    Args:
        modes: Configured operating modes.  Must be non-empty.
        mode: 0-based index into ``modes``.

    Returns:
        The selected :class:`ADCMode`.

    Raises:
        ValueError: If ``modes`` is empty or ``mode`` is out of range.
    """
    if not modes:
        raise ValueError("ADC config must declare at least one mode")
    if not (0 <= mode < len(modes)):
        raise ValueError(f"mode index {mode} out of range for {len(modes)} configured modes")
    return modes[mode]


def quantize_with_jitter(
    signal: Tensor,
    boundaries: Tensor,
    *,
    lsb: float,
    training: bool,
) -> Tensor:
    """Floor-bucketize with optional uniform jitter scaled to ``lsb``.

    A thin wrapper that fixes the dtype to ``int16`` and centralises
    the LSB-jitter convention used across every multi-mode subclass.

    Args:
        signal: Analog input.
        boundaries: Floor-style threshold tensor.
        lsb: Bin width (used to size the stochastic jitter).
        training: ``module.training`` flag.

    Returns:
        ``int16`` code tensor of shape ``signal.shape``.
    """
    return floor_bucketize(
        signal,
        boundaries,
        out_dtype=torch.int16,
        training=training,
        lsb=lsb,
    )
