"""Unit tests for :class:`Offset1T1RXbar`'s batch-chunking scheduler.

Covers the chunking contract in ``vec_mat_mul``:
  - ``batch_chunk_size <= 0`` disables chunking entirely.
  - any positive ``batch_chunk_size`` must yield bit-exact ADC codes vs the
    unchunked path under deterministic policies.
  - non-divisible batches (remainder chunk) work correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurox.analog.adc import AdcOperationPoint
from neurox.tools.xbar_adc._sampling import (
    build_offset_1t1r_xbar_all_off,
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "presets" / "xbar" / "1t1r_28nm.toml"


def _build(chunk_size: int, device: torch.device, *, inst: int = 4):
    return build_offset_1t1r_xbar_all_off(
        XBAR_CONFIG,
        device=device,
        inst_shape=(inst,),
        batch_chunk_size=chunk_size,
    )


def _run(chunk_size: int, device: torch.device, *, x_batch: int = 8) -> torch.Tensor:
    xbar = _build(chunk_size, device)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    w = next(iter(sample_w(distribution, xbar, n=xbar._inst_shape[0], batch_w=xbar._inst_shape[0],
                           device=device, generator=g)))
    xbar.program(w)
    x = next(iter(sample_x_batches(distribution, xbar, n_total=x_batch,
                                    batch_size=x_batch, device=device, generator=g)))
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    return xbar.vec_mat_mul(x.unsqueeze(-2), adc_operation_point=op)


@pytest.fixture(scope="module")
def fixture_config():
    """Skip the suite if the chip-fixture TOML isn't present."""
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_chunk_zero_matches_unchunked(fixture_config, device):
    """A chunk_size <= 0 must skip the scheduler entirely."""
    codes_disabled = _run(0, device)
    codes_negative = _run(-1, device)
    assert torch.equal(codes_disabled, codes_negative)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8])
def test_chunk_bit_exact(fixture_config, device, chunk_size):
    """Chunked path is bit-exact with unchunked under deterministic policy."""
    codes_full = _run(0, device, x_batch=8)
    codes_chunked = _run(chunk_size, device, x_batch=8)
    assert torch.equal(codes_full, codes_chunked), (
        f"chunk_size={chunk_size} produced different codes "
        f"(max abs diff: {(codes_full.float() - codes_chunked.float()).abs().max().item():.4e})"
    )


def test_chunk_size_larger_than_batch_is_noop(fixture_config, device):
    """``chunk_size > batch`` must early-exit through the unchunked path."""
    codes_full = _run(0, device, x_batch=8)
    codes_big = _run(100, device, x_batch=8)
    assert torch.equal(codes_full, codes_big)


def test_remainder_chunk(fixture_config, device):
    """Non-divisible batches (last chunk smaller than chunk_size) must work."""
    # 8 samples / chunk=3 → 3+3+2 chunks.
    codes_full = _run(0, device, x_batch=8)
    codes_remainder = _run(3, device, x_batch=8)
    assert torch.equal(codes_full, codes_remainder)
