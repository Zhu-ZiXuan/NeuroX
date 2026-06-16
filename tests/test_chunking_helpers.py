"""Unit + targeted-integration tests for the chunking helpers.

`tests/test_xbar_chunking.py` covers the chunking scheduler at the
``vec_mat_mul`` integration level on a single-axis ``inst_shape=(N,)``
fixture. This file goes lower:

- Direct assertions on ``iter_chunks`` / ``reassemble_chunks`` to lock the
  flat C-order multi-coords + flat-index ordering that the scatter path
  relies on. A single ``solve_chunk_size`` budget partitions the broadcast
  leading into contiguous slices regardless of which leading axes are
  serial (A) or inst (B), so chunk boundaries cross axis boundaries
  freely — the tests assert exactly that.
- ``classify_leading_positions`` — still used by ``cim_read`` to count the
  serial per-op latency multiplicity (A subset), not for chunking.
- Degenerate ``leading=()`` (xbar with empty ``inst_shape`` and a 1-D
  input vector) — exercises ``reassemble_chunks``'s short-circuit path
  for 0-D / 1-D payloads where ``torch.cat`` is ill-defined.
- A macro-shaped ``(M, Sa, Sw, Tc, Tr)`` integration run on
  ``Offset1T1RXbar`` to exercise chunking under realistic leading rank.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.tools.xbar_adc._sampling import (
    build_offset_1t1r_xbar_all_off,
    load_distribution,
    make_generator,
    sample_w,
    sample_x_batches,
)
from neurox.xbar import Offset1T1RXbar
from neurox.xbar._1t1r._chunking import (
    ChunkSpec,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# classify_leading_positions
# ---------------------------------------------------------------------------


def test_classify_x_real_g_placeholder_is_a() -> None:
    # leading rank 2; x has real position 0, g has size-1 placeholder
    a, b = classify_leading_positions(x_shape=(4, 1, 1, 8), g_shape=(1, 1, 5, 8), leading_rank=2)
    assert a == (0,)
    assert b == ()


def test_classify_g_real_is_b() -> None:
    a, b = classify_leading_positions(x_shape=(1, 1, 1, 8), g_shape=(2, 3, 5, 8), leading_rank=2)
    assert a == ()
    assert b == (0, 1)


def test_classify_both_real_position_goes_to_b() -> None:
    # Matched positions (both > 1) — e.g. Tc, which is matched on x and g
    # sides simultaneously — must land in B (parallel inst), not A.
    a, b = classify_leading_positions(x_shape=(2, 1, 8), g_shape=(2, 5, 8), leading_rank=1)
    assert a == ()
    assert b == (0,)


def test_classify_degenerate_both_one_is_omitted() -> None:
    a, b = classify_leading_positions(x_shape=(1, 1, 1, 8), g_shape=(1, 1, 5, 8), leading_rank=2)
    assert a == ()
    assert b == ()


# ---------------------------------------------------------------------------
# iter_chunks — flat C-order contiguous slices of a known (2, 3) leading
# ---------------------------------------------------------------------------


def _materialise_chunks(leading: tuple[int, ...], chunk_size: int) -> list[ChunkSpec]:
    return list(iter_chunks(leading=leading, chunk_size=chunk_size, device=CPU))


def test_iter_chunks_single_block_when_size_zero() -> None:
    chunks = _materialise_chunks(leading=(2, 3), chunk_size=0)
    assert len(chunks) == 1
    spec = chunks[0]
    assert spec.chunk_size == 6
    # Multi-coords match torch.unravel_index over the flattened leading.
    assert torch.equal(spec.flat_global_idx, torch.arange(6))
    assert torch.equal(spec.multi_coords[0], torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(spec.multi_coords[1], torch.tensor([0, 1, 2, 0, 1, 2]))


def test_iter_chunks_contiguous_c_order_slices() -> None:
    # leading=(2,3) → 6 instances; chunk_size=2 → 3 contiguous flat slices
    # [0,1], [2,3], [4,5]. The single budget is axis-agnostic: chunk 1
    # straddles the boundary between leading dim 0's row 0 and row 1.
    chunks = _materialise_chunks(leading=(2, 3), chunk_size=2)
    assert len(chunks) == 3

    assert torch.equal(chunks[0].flat_global_idx, torch.tensor([0, 1]))
    assert torch.equal(chunks[0].multi_coords[0], torch.tensor([0, 0]))
    assert torch.equal(chunks[0].multi_coords[1], torch.tensor([0, 1]))

    # chunk 1 crosses the leading-dim-0 boundary: flat 2 → (0, 2), flat 3 → (1, 0).
    assert torch.equal(chunks[1].flat_global_idx, torch.tensor([2, 3]))
    assert torch.equal(chunks[1].multi_coords[0], torch.tensor([0, 1]))
    assert torch.equal(chunks[1].multi_coords[1], torch.tensor([2, 0]))

    assert torch.equal(chunks[2].flat_global_idx, torch.tensor([4, 5]))
    assert torch.equal(chunks[2].multi_coords[0], torch.tensor([1, 1]))
    assert torch.equal(chunks[2].multi_coords[1], torch.tensor([1, 2]))


def test_iter_chunks_remainder_chunk() -> None:
    # 6 instances, chunk_size=4 → [0,1,2,3] then a smaller [4,5] remainder.
    chunks = _materialise_chunks(leading=(2, 3), chunk_size=4)
    assert len(chunks) == 2
    assert chunks[0].chunk_size == 4
    assert chunks[1].chunk_size == 2
    assert torch.equal(chunks[0].flat_global_idx, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(chunks[1].flat_global_idx, torch.tensor([4, 5]))


def test_iter_chunks_empty_leading_yields_one_chunk() -> None:
    chunks = _materialise_chunks(leading=(), chunk_size=0)
    assert len(chunks) == 1
    spec = chunks[0]
    assert spec.chunk_size == 1
    assert spec.multi_coords == ()
    assert torch.equal(spec.flat_global_idx, torch.tensor([0]))


def test_iter_chunks_chunk_size_clipped_to_extent() -> None:
    # Over-sized chunk_size clips down to a single chunk covering everything.
    chunks = _materialise_chunks(leading=(2, 3), chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_size == 6
    assert torch.equal(chunks[0].flat_global_idx, torch.arange(6))


# ---------------------------------------------------------------------------
# reassemble_chunks — scatter correctness incl. degenerate leading
# ---------------------------------------------------------------------------


def test_reassemble_empty_leading_returns_chunk_unchanged() -> None:
    # leading=() → single chunk carries the full trailing payload.
    payload = torch.tensor([1.5, 2.5, 3.5])
    out = reassemble_chunks(
        chunks=[payload],
        chunk_global_indices=[torch.tensor([0])],
        leading=(),
        trailing=(3,),
    )
    assert torch.equal(out, payload)


def test_reassemble_empty_leading_scalar_trailing() -> None:
    # leading=() with trailing=() — 0-D payload. Pre-fix this crashed in
    # torch.cat(..., dim=0); now it short-circuits.
    payload = torch.tensor(7.0)
    out = reassemble_chunks(
        chunks=[payload],
        chunk_global_indices=[torch.tensor([0])],
        leading=(),
        trailing=(),
    )
    assert torch.equal(out, payload)
    assert out.ndim == 0


def test_reassemble_scatter_recovers_canonical_order() -> None:
    # Feed reassemble in a non-canonical chunk order and verify the
    # scatter places each row back at its global flat index.
    leading = (2, 3)
    trailing = (4,)
    # Build a (6, 4) "reference" canonical tensor, then split it into
    # arbitrary chunk groups and feed them out of order.
    canonical = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    chunk_a = canonical[[5, 2]]
    chunk_b = canonical[[0, 1, 4]]
    chunk_c = canonical[[3]]
    out = reassemble_chunks(
        chunks=[chunk_a, chunk_b, chunk_c],
        chunk_global_indices=[
            torch.tensor([5, 2]),
            torch.tensor([0, 1, 4]),
            torch.tensor([3]),
        ],
        leading=leading,
        trailing=trailing,
    )
    expected = canonical.reshape(2, 3, 4)
    assert torch.equal(out, expected)


def test_reassemble_round_trip_against_iter_chunks() -> None:
    # End-to-end: iterate chunks of a known (leading, trailing) source,
    # gather them, scatter back, and confirm the result equals the
    # source via plain advanced-indexing. chunk_size=5 over 12 instances
    # forces two remainder-straddling boundaries.
    leading = (3, 4)
    trailing = (5,)
    src = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(*leading, *trailing)
    src_flat = src.reshape(-1, *trailing)

    chunks, idxs = [], []
    for spec in iter_chunks(leading=leading, chunk_size=5, device=CPU):
        chunks.append(src_flat[spec.flat_global_idx])
        idxs.append(spec.flat_global_idx)

    rebuilt = reassemble_chunks(chunks, idxs, leading, trailing)
    assert torch.equal(rebuilt, src)


# ---------------------------------------------------------------------------
# Multi-leading-dim integration — (M, Sa, Sw, Tc, Tr) macro shape
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_config() -> Iterator[Path]:
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else CPU


def _build_multi_inst(
    chunk_size: int,
    device: torch.device,
    *,
    inst_shape: tuple[int, ...],
) -> Offset1T1RXbar:
    return build_offset_1t1r_xbar_all_off(
        XBAR_CONFIG,
        device=device,
        inst_shape=inst_shape,
        solve_chunk_size=chunk_size,
    )


def _run_multi_leading(
    chunk_size: int,
    device: torch.device,
    *,
    inst_shape: tuple[int, ...] = (2, 1, 2),  # (Sw, Tc, Tr)
    m: int = 2,
    sa: int = 3,
) -> torch.Tensor:
    """Drive Offset1T1RXbar with a (M, Sa, *inst_shape) leading shape.

    Total leading rank = ``2 + len(inst_shape)`` — 5 here, matching the
    real macro layout (``M, Sa, Sw, Tc, Tr``). The single ``solve_chunk_size``
    budget partitions the flattened leading (``prod = M·Sa·Sw·Tc·Tr``) into
    contiguous slices that cross every axis boundary freely.
    """
    xbar = _build_multi_inst(chunk_size, device, inst_shape=inst_shape)
    distribution = load_distribution(None, xbar)
    g = make_generator(0, device)
    # Number of physical weight batches = product of inst dims (one weight
    # tensor per fab instance). sample_w returns a flat (n, col, digit, row)
    # tensor; reshape into the full ``_w_layout_shape``.
    n_inst = int(torch.tensor(inst_shape).prod().item())
    w_flat = next(iter(sample_w(distribution, xbar, n=n_inst, batch_w=n_inst, device=device, generator=g)))
    w = w_flat.reshape(*xbar._w_layout_shape)
    xbar.program(w)
    # x shape: (M, Sa, *inst_ones, row_num) — primitive trailing is
    # exactly [row_num] per vec_mat_mul's public contract; core's
    # internal unsqueeze(-2) adds the WL-fanout slot. The inst-broadcast
    # ones are real leading dims (M, Sa are x-side; inst_ones are
    # broadcast-against-inst placeholders), giving leading rank ≥
    # len(inst_shape) + 2.
    row_num = xbar.row_num
    x_flat = next(
        iter(sample_x_batches(distribution, xbar, n_total=m * sa, batch_size=m * sa, device=device, generator=g))
    )
    inst_ones = (1,) * len(inst_shape)
    x = x_flat.reshape(m, sa, *inst_ones, row_num)
    op = AdcOperationPoint(adc_mode=0, adc_bits=8)
    result = xbar.vec_mat_mul(x, adc_operation_point=op)
    assert isinstance(result, Tensor)
    return result


def test_macro_shape_leading_single_block(fixture_config: Path, device: torch.device) -> None:
    out = _run_multi_leading(0, device)
    # Leading is (M, Sa) x-side + inst_shape. ``w_digit_count`` is
    # already flattened into ``phys_col`` by ``Offset1T1RXbar.program``
    # before reaching core, so it is not a leading dim here. Just
    # confirm we have a meaningfully multi-dim leading and the result
    # tensor has the expected outer shape.
    assert out.ndim >= 6
    assert out.shape[0] == 2
    assert out.shape[1] == 3


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 7, 24, 100])
def test_macro_shape_leading_chunked_bit_exact(fixture_config: Path, device: torch.device, chunk_size: int) -> None:
    # Leading total = M·Sa·Sw·Tc·Tr = 2·3·2·1·2 = 24. The chunk sizes span
    # exact divisors (2), remainder-straddling values (5, 7), exact total
    # (24), and over-extent (100) — all must be bit-exact vs single-block.
    full = _run_multi_leading(0, device)
    chunked = _run_multi_leading(chunk_size, device)
    assert torch.equal(full, chunked), f"solve_chunk_size={chunk_size} perturbed the result"
