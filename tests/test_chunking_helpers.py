"""Unit + targeted-integration tests for the chunking helpers.

`tests/test_xbar_chunking.py` covers the chunking scheduler at the
``vec_mat_mul`` integration level on a single-axis ``inst_shape=(N,)``
fixture. This file goes lower:

- Direct assertions on ``iter_chunks`` / ``reassemble_chunks`` /
  ``_row_major_strides`` to lock the multi-coords + flat-index ordering
  that the scatter path relies on.
- Degenerate ``leading=()`` (xbar with empty ``inst_shape`` and a 1-D
  input vector) — exercises ``reassemble_chunks``'s short-circuit path
  for 0-D / 1-D payloads where ``torch.cat`` is ill-defined.
- A macro-shaped ``(M, Sa, Sw, Tc, Tr)`` integration run on
  ``Offset1T1RXbar`` to exercise mixed A/B chunking under realistic
  leading rank.
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
from neurox.xbar._1t1r._chunking import (
    _row_major_strides,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# _row_major_strides
# ---------------------------------------------------------------------------


def test_row_major_strides_empty() -> None:
    assert _row_major_strides(()) == []


def test_row_major_strides_single() -> None:
    assert _row_major_strides((4,)) == [1]


def test_row_major_strides_rank3() -> None:
    # shape (2, 3, 4) → strides (12, 4, 1) for C-order
    assert _row_major_strides((2, 3, 4)) == [12, 4, 1]


def test_row_major_strides_match_torch_unravel() -> None:
    # The C-order stride contract is the same one ``torch.unravel_index``
    # uses; cross-check on a non-trivial shape.
    shape = (3, 5, 7)
    strides = _row_major_strides(shape)
    flat = torch.arange(3 * 5 * 7)
    multi = torch.unravel_index(flat, shape)
    rebuilt = sum(int(strides[d]) * multi[d] for d in range(len(shape)))
    assert torch.equal(rebuilt, flat)


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
# iter_chunks — sequence + ordering on a known (A=2, B=3) layout
# ---------------------------------------------------------------------------


def _materialise_chunks(leading, a_positions, b_positions, cx, ci):
    return list(
        iter_chunks(
            leading=leading,
            a_positions=a_positions,
            b_positions=b_positions,
            chunk_size_x=cx,
            chunk_size_inst=ci,
            device=CPU,
        )
    )


def test_iter_chunks_single_block_when_both_sizes_zero() -> None:
    chunks = _materialise_chunks(leading=(2, 3), a_positions=(0,), b_positions=(1,), cx=0, ci=0)
    assert len(chunks) == 1
    spec = chunks[0]
    assert spec.chunk_size == 6
    # Multi-coords match torch.unravel_index over the flattened leading.
    assert torch.equal(spec.flat_global_idx, torch.arange(6))
    assert torch.equal(spec.multi_coords[0], torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(spec.multi_coords[1], torch.tensor([0, 1, 2, 0, 1, 2]))


def test_iter_chunks_a_outer_b_inner_ordering() -> None:
    # leading=(2,3); A at position 0, B at position 1; cx=1, ci=2.
    # → 4 chunks: (a=0,b={0,1}), (a=0,b={2}), (a=1,b={0,1}), (a=1,b={2}).
    chunks = _materialise_chunks(leading=(2, 3), a_positions=(0,), b_positions=(1,), cx=1, ci=2)
    assert len(chunks) == 4

    # chunk 0: a=0, b=[0,1] → flat=[0, 1]
    assert torch.equal(chunks[0].flat_global_idx, torch.tensor([0, 1]))
    assert torch.equal(chunks[0].multi_coords[0], torch.tensor([0, 0]))
    assert torch.equal(chunks[0].multi_coords[1], torch.tensor([0, 1]))

    # chunk 1: a=0, b=[2] (remainder) → flat=[2]
    assert torch.equal(chunks[1].flat_global_idx, torch.tensor([2]))
    assert torch.equal(chunks[1].multi_coords[0], torch.tensor([0]))
    assert torch.equal(chunks[1].multi_coords[1], torch.tensor([2]))

    # chunk 2: a=1, b=[0,1] → flat=[3, 4]
    assert torch.equal(chunks[2].flat_global_idx, torch.tensor([3, 4]))
    assert torch.equal(chunks[2].multi_coords[0], torch.tensor([1, 1]))
    assert torch.equal(chunks[2].multi_coords[1], torch.tensor([0, 1]))

    # chunk 3: a=1, b=[2] → flat=[5]
    assert torch.equal(chunks[3].flat_global_idx, torch.tensor([5]))
    assert torch.equal(chunks[3].multi_coords[0], torch.tensor([1]))
    assert torch.equal(chunks[3].multi_coords[1], torch.tensor([2]))


def test_iter_chunks_empty_leading_yields_one_chunk() -> None:
    chunks = _materialise_chunks(leading=(), a_positions=(), b_positions=(), cx=0, ci=0)
    assert len(chunks) == 1
    spec = chunks[0]
    assert spec.chunk_size == 1
    assert spec.multi_coords == ()
    assert torch.equal(spec.flat_global_idx, torch.tensor([0]))


def test_iter_chunks_chunk_size_clipped_to_extent() -> None:
    # Over-sized cx clips down; the per-chunk multi-coords still cover
    # every position once.
    chunks = _materialise_chunks(leading=(2, 3), a_positions=(0,), b_positions=(1,), cx=100, ci=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_size == 6


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
    # source via plain advanced-indexing.
    leading = (3, 4)
    trailing = (5,)
    src = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(*leading, *trailing)
    src_flat = src.reshape(-1, *trailing)

    chunks, idxs = [], []
    for spec in iter_chunks(
        leading=leading,
        a_positions=(0,),
        b_positions=(1,),
        chunk_size_x=2,
        chunk_size_inst=3,
        device=CPU,
    ):
        chunks.append(src_flat[spec.flat_global_idx])
        idxs.append(spec.flat_global_idx)

    rebuilt = reassemble_chunks(chunks, idxs, leading, trailing)
    assert torch.equal(rebuilt, src)


# ---------------------------------------------------------------------------
# Multi-leading-dim integration — (M, Sa, Sw, Tc, Tr) macro shape
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_config():
    if not XBAR_CONFIG.is_file():
        pytest.skip(f"missing test fixture: {XBAR_CONFIG}")
    yield XBAR_CONFIG


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0") if torch.cuda.is_available() else CPU


def _build_multi_inst(
    chunk_x: int,
    chunk_inst: int,
    device: torch.device,
    *,
    inst_shape: tuple[int, ...],
):
    return build_offset_1t1r_xbar_all_off(
        XBAR_CONFIG,
        device=device,
        inst_shape=inst_shape,
        solve_chunk_size_x=chunk_x,
        solve_chunk_size_inst=chunk_inst,
    )


def _run_multi_leading(
    chunk_x: int,
    chunk_inst: int,
    device: torch.device,
    *,
    inst_shape: tuple[int, ...] = (2, 1, 2),  # (Sw, Tc, Tr)
    m: int = 2,
    sa: int = 3,
) -> torch.Tensor:
    """Drive Offset1T1RXbar with a (M, Sa, *inst_shape) leading shape.

    Total leading rank = ``2 + len(inst_shape)`` — 5 here, matching the
    real macro layout (``M, Sa, Sw, Tc, Tr``).
    """
    xbar = _build_multi_inst(chunk_x, chunk_inst, device, inst_shape=inst_shape)
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
    return xbar.vec_mat_mul(x, adc_operation_point=op)


def test_macro_shape_leading_single_block(fixture_config, device) -> None:
    out = _run_multi_leading(0, 0, device)
    # Leading is (M, Sa) x-side + inst_shape. ``w_digit_count`` is
    # already flattened into ``phys_col`` by ``Offset1T1RXbar.program``
    # before reaching core, so it is not a leading dim here. Just
    # confirm we have a meaningfully multi-dim leading and the result
    # tensor has the expected outer shape.
    assert out.ndim >= 6
    assert out.shape[0] == 2
    assert out.shape[1] == 3


def test_macro_shape_leading_a_chunked_bit_exact(fixture_config, device) -> None:
    full = _run_multi_leading(0, 0, device)
    chunked = _run_multi_leading(2, 0, device)
    assert torch.equal(full, chunked)


def test_macro_shape_leading_b_chunked_bit_exact(fixture_config, device) -> None:
    full = _run_multi_leading(0, 0, device)
    chunked = _run_multi_leading(0, 2, device)
    assert torch.equal(full, chunked)


def test_macro_shape_leading_ab_mixed_remainder(fixture_config, device) -> None:
    # Use M=2, Sa=3 (A-side total 6) with cx=4 → remainder 2.
    # inst total = 2*1*2 = 4 with ci=3 → remainder 1.
    full = _run_multi_leading(0, 0, device)
    chunked = _run_multi_leading(4, 3, device)
    assert torch.equal(full, chunked)
