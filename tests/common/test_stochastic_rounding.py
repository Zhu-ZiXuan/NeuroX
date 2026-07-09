"""Tests for stochastic rounding helpers shared by every quantizer.

Three properties under test:

1. **Eval determinism** — when ``training=False``, the helper reduces
   to a plain floor / floor-bucketize and produces the same output
   every call.
2. **Train randomness** — when ``training=True``, repeated calls
   produce a distribution of outputs around the underlying real-valued
   quotient.
3. **Train unbiasedness** — the mean of the stochastic output equals
   the underlying real-valued quotient (within Monte-Carlo noise).
"""

from __future__ import annotations

import math

import torch

from neurox.common.quant import (
    floor_bucketize,
    stochastic_floor_div,
    stochastic_floor_to_int,
)


def test_eval_floor_div_deterministic() -> None:
    """Eval mode = plain right-shift; output is deterministic."""
    x = torch.tensor([7, 8, 15, 16, 31], dtype=torch.int64)
    y1 = stochastic_floor_div(x, 3, training=False)
    y2 = stochastic_floor_div(x, 3, training=False)
    assert torch.equal(y1, y2)
    expected = x >> 3
    assert torch.equal(y1, expected)


def test_train_floor_div_unbiased() -> None:
    """Train mode jitter is unbiased; sample mean → real quotient.

    Use an integer-aligned x so the real quotient is exact and avoid
    truncation bias from ``int(real * denom)``.
    """
    rshift = 4
    denom = 1 << rshift
    # x = 5; real value = 5/16 = 0.3125.
    x_int = 5
    x_real = x_int / denom
    x = torch.full((100_000,), x_int, dtype=torch.int64)
    samples = stochastic_floor_div(x, rshift, training=True).float()
    assert math.isclose(samples.mean().item(), x_real, abs_tol=2e-2)
    # The set of observed values is exactly {0, 1} for x_real ∈ (0, 1).
    unique = torch.unique(samples).tolist()
    assert sorted(unique) == [0, 1]


def test_eval_floor_to_int_deterministic() -> None:
    sig = torch.tensor([0.0, 0.49, 0.5, 0.99, 1.0], dtype=torch.float32)
    scale = 10.0  # codes per signal unit (1 / lsb where lsb = 0.1)
    y1 = stochastic_floor_to_int(sig, scale, out_dtype=torch.int16, training=False)
    y2 = stochastic_floor_to_int(sig, scale, out_dtype=torch.int16, training=False)
    assert torch.equal(y1, y2)
    assert y1.tolist() == [0, 4, 5, 9, 10]


def test_train_floor_to_int_unbiased() -> None:
    sig = torch.full((50_000,), 0.275, dtype=torch.float32)
    scale = 10.0  # codes per signal unit (1 / lsb where lsb = 0.1)
    samples = stochastic_floor_to_int(
        sig,
        scale,
        out_dtype=torch.int32,
        training=True,
    ).float()
    # Real value 2.75 → expected mean ≈ 2.75.
    assert math.isclose(samples.mean().item(), 2.75, abs_tol=5e-2)


def test_floor_bucketize_eval() -> None:
    boundaries = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    sig = torch.tensor([0.0, 0.15, 0.25, 0.35, 0.45], dtype=torch.float32)
    code = floor_bucketize(sig, boundaries, out_dtype=torch.int16, training=False, lsb=0.1)
    assert code.tolist() == [0, 1, 2, 3, 4]


def test_training_flag_drives_jitter() -> None:
    """``training=True`` enables jitter; ``training=False`` disables it."""
    x = torch.full((1000,), 5, dtype=torch.int64)
    eval_out = stochastic_floor_div(x, 3, training=False)
    train_out = stochastic_floor_div(x, 3, training=True)
    assert torch.equal(eval_out, x >> 3)
    # Train mode produces a distribution; eval mode is deterministic.
    assert torch.unique(train_out).numel() >= 2
