"""Tests for chunk construction, tensor-tree slicing, folding, and execution.

Covers the chunk coordinates and tail padding, the collapse-index-re-expand
slicing rule (asserted on storage size, not only values), the declared-leading
check, and the preallocating fold that writes each result into its contiguous
interval.

The executor section injects a stub unary function that records exactly what
it was handed and pins four laws:

  * OPERAND-TREE LAW: every tensor field of the injected operand dataclass,
    including nested snaps and a bare tensor field, is sliced together.
  * DECLARED-LEADING LAW: the caller states the leading and every sliced
    operand carries it; the injected function sees one fixed chunk shape no
    larger than the complete leading, and the tail chunk is padded to it.
  * FOLD LAW: what comes back is the result type at the full leading, equal to
    a single whole-leading run element for element.
  * PASS-THROUGH LAW: with no leading, the operand tree reaches the function
    untouched and the single result is returned as it stands.

Additional checks guard the declaration itself: whether nested in a snap or
stored directly, an operand tensor that does not carry the stated leading is
rejected before any chunk runs.

The last section drives the whole layer through a real 1T1R array and pins
the two laws the fold exists for:

  * chunk size is a memory knob — the port state and the billed energy are
    bit-identical across chunk sizes, the un-chunked solve included,
  * what survives a chunk scales with the columns, not with the cells: the
    state retained across chunk boundaries is flat in `row_num`.

The retained state is measured on CPU by walking every Python-reachable
tensor at each chunk boundary and summing its storage once per data pointer.
"""

import gc
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox import Profiler, stamp_names
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
    XbarArray1t1rScanMode,
    XbarArray1t1rSteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.primitive.xbar.solver import ColBlColSlSolverConfig, execute_chunked
from neurox.primitive.xbar.solver.chunking import _Chunk, _iter_chunks, _ResultFold, _slice_tensor, _slice_tensor_tree

_COL = 4
_ROW = 5
_CPU = torch.device("cpu")


@dataclass(frozen=True)
class _DeviceSnap:
    g__uS: Tensor


@dataclass(frozen=True)
class _GridSnap:
    buffer: Tensor
    wl_drive: Tensor
    partial: Tensor
    full: Tensor
    device: _DeviceSnap
    t_elapsed: float


@dataclass(frozen=True)
class _ColumnSnap:
    v_ref__V: Tensor
    r_out__MOhm: Tensor


@dataclass(frozen=True)
class _CellSnap:
    v_wl__V: Tensor
    g__uS: Tensor


@dataclass(frozen=True)
class _CellDcop:
    i__uA: Tensor


@dataclass(frozen=True)
class _Dcop:
    i_bl_driver__uA: Tensor
    v_bl_node__V: Tensor
    cell: _CellDcop
    label: str


@dataclass(frozen=True)
class _Result:
    """Per-chunk result carrying one optional field."""

    i_port__uA: Tensor
    energy__fJ: Tensor | None


@dataclass(frozen=True)
class _PortResult:
    """What survives one run: the port state and one folded row sum."""

    i_port__uA: Tensor
    wl_sum__V: Tensor


def _elements(t: Tensor) -> int:
    """Storage elements a tensor actually occupies."""
    return t.untyped_storage().size() // t.element_size()


def _chunks(leading_shape: tuple[int, ...], chunk_size: int) -> list[_Chunk]:
    return list(
        _iter_chunks(
            leading_shape=leading_shape,
            chunk_size=chunk_size,
            reference=torch.empty((), device=_CPU),
        )
    )


def _grid_snap(leading: tuple[int, ...]) -> _GridSnap:
    """Build one snap covering all four rows of the broadcast table.

    Every field carries the full leading, as the chunking layer requires;
    the four rows differ in which of those axes are stride-0 views.
    """
    inner = leading[-1]
    return _GridSnap(
        # No real leading: a programmed cell buffer, held across the call.
        buffer=torch.randn(_COL, _ROW).expand(*leading, _COL, _ROW),
        # Stride-0 on the column axis: one word line per row, held across columns.
        wl_drive=torch.randn(*leading, 1, _ROW).expand(*leading, _COL, _ROW),
        # Partially broadcast: only the inner leading axis is real.
        partial=torch.randn(1, inner, _COL, _ROW).expand(*leading, _COL, _ROW),
        # Fully real.
        full=torch.randn(*leading, _COL, _ROW),
        device=_DeviceSnap(g__uS=torch.randn(*leading, _COL, _ROW)),
        t_elapsed=0.0,
    )


def test_positive_chunk_size_pads_only_solver_coordinates() -> None:
    chunks = _chunks((5,), 3)

    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 3), (3, 5)]
    torch.testing.assert_close(chunks[0].coords[0], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(chunks[1].coords[0], torch.tensor([3, 4, 4]))


def test_chunk_size_is_capped_by_the_workload() -> None:
    [chunk] = _chunks((2,), 5)

    assert (chunk.start, chunk.stop) == (0, 2)
    torch.testing.assert_close(chunk.coords[0], torch.tensor([0, 1]))


def test_slicer_storage_follows_the_broadcast_table() -> None:
    leading = (2, 3)
    chunk_size = 4
    snap = _grid_snap(leading)
    [chunk, _tail] = _chunks(leading, chunk_size)

    sliced = _slice_tensor_tree(snap, coords=chunk.coords, leading_shape=leading)

    # No real leading: one chunk position, broadcasting against the rest.
    assert sliced.buffer.shape == (1, _COL, _ROW)
    assert _elements(sliced.buffer) == _COL * _ROW

    # Stride-0 on the column axis: one value per (chunk position, row).
    assert sliced.wl_drive.shape == (chunk_size, _COL, _ROW)
    assert _elements(sliced.wl_drive) == chunk_size * _ROW

    # Partially broadcast: indexed by the inner coordinates only.
    assert sliced.partial.shape == (chunk_size, _COL, _ROW)
    assert _elements(sliced.partial) == chunk_size * _COL * _ROW

    # Fully real: materialised, unavoidable.
    assert sliced.full.shape == (chunk_size, _COL, _ROW)
    assert _elements(sliced.full) == chunk_size * _COL * _ROW


def test_slicer_collapses_a_stride_zero_column_axis() -> None:
    leading = (6,)
    chunk_size = 4
    wl_drive = torch.randn(6, 1, _ROW).expand(6, _COL, _ROW)
    snap = _DeviceSnap(g__uS=wl_drive)
    [chunk, _tail] = _chunks(leading, chunk_size)

    sliced = _slice_tensor_tree(snap, coords=chunk.coords, leading_shape=leading)

    assert sliced.g__uS.shape == (chunk_size, _COL, _ROW)
    assert _elements(sliced.g__uS) == chunk_size * _ROW
    assert _elements(wl_drive[chunk.coords]) == chunk_size * _COL * _ROW
    torch.testing.assert_close(sliced.g__uS, wl_drive[chunk.coords])


def test_slicer_values_match_a_dense_gather() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    [first, tail] = _chunks(leading, 4)

    for chunk in (first, tail):
        coords = chunk.coords
        sliced = _slice_tensor_tree(snap, coords=coords, leading_shape=leading)

        dense_buffer = snap.buffer[coords]
        dense_wl = snap.wl_drive[coords]
        dense_partial = snap.partial[coords]
        dense_full = snap.full[coords]
        torch.testing.assert_close(sliced.buffer.expand_as(dense_buffer), dense_buffer)
        torch.testing.assert_close(sliced.wl_drive, dense_wl)
        torch.testing.assert_close(sliced.partial, dense_partial)
        torch.testing.assert_close(sliced.full, dense_full)


def test_slicer_recurses_into_nested_snaps_and_keeps_plain_fields() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    [chunk, _tail] = _chunks(leading, 4)

    sliced = _slice_tensor_tree(snap, coords=chunk.coords, leading_shape=leading)

    torch.testing.assert_close(sliced.device.g__uS, snap.device.g__uS[chunk.coords])
    assert sliced.t_elapsed == 0.0
    assert type(sliced) is _GridSnap
    # The input snap is left as it was.
    assert snap.full.shape == (2, 3, _COL, _ROW)


def test_slicer_reads_the_trailing_block_off_the_leading_it_is_given() -> None:
    leading = (6,)
    chunk_size = 4
    snap = _ColumnSnap(
        v_ref__V=torch.randn(6, _COL),
        # A constant, stated at the call's shape as every snap field is.
        r_out__MOhm=torch.zeros(()).expand(6, _COL),
    )
    [chunk, _tail] = _chunks(leading, chunk_size)

    sliced = _slice_tensor_tree(snap, coords=chunk.coords, leading_shape=leading)

    # One trailing axis behind the leading, and it is the one that survives.
    assert sliced.v_ref__V.shape == (chunk_size, _COL)
    assert _elements(sliced.v_ref__V) == chunk_size * _COL
    # A field that is stride-0 throughout costs one element and broadcasts back.
    assert sliced.r_out__MOhm.shape == (1, _COL)
    assert _elements(sliced.r_out__MOhm) == 1


def test_slicer_keeps_the_chunk_axis_on_a_singleton_leading() -> None:
    leading = (1,)
    snap = _ColumnSnap(v_ref__V=torch.randn(1, _COL), r_out__MOhm=torch.zeros(()).expand(1, _COL))
    [chunk] = _chunks(leading, 3)

    sliced = _slice_tensor_tree(snap, coords=chunk.coords, leading_shape=leading)

    assert sliced.v_ref__V.shape == (1, _COL)


def test_a_bare_tensor_slices_by_the_same_rule_as_a_snap_field() -> None:
    """The dataclass is a walk, not a precondition: the rule is the tensor's."""
    leading = (2, 3)
    chunk_size = 4
    wl_drive = torch.randn(*leading, 1, _ROW).expand(*leading, _COL, _ROW)
    [chunk, _tail] = _chunks(leading, chunk_size)

    bare = _slice_tensor(wl_drive, coords=chunk.coords, leading_shape=leading)
    wrapped = _slice_tensor_tree(_DeviceSnap(g__uS=wl_drive), coords=chunk.coords, leading_shape=leading)

    # Same shape, same values, and the same collapsed storage.
    assert bare.shape == (chunk_size, _COL, _ROW)
    assert _elements(bare) == chunk_size * _ROW
    torch.testing.assert_close(bare, wrapped.g__uS)
    assert _elements(bare) == _elements(wrapped.g__uS)


def _solve(snap: _GridSnap, coords: tuple[Tensor, ...], leading: tuple[int, ...]) -> _Dcop:
    """Stand-in solver: one fixed-shape pass over the chunk."""
    sliced = _slice_tensor_tree(snap, coords=coords, leading_shape=leading)
    v_bl_node__V = sliced.buffer + sliced.wl_drive + sliced.partial + sliced.full
    return _Dcop(
        i_bl_driver__uA=v_bl_node__V.sum(dim=-1),
        v_bl_node__V=v_bl_node__V,
        cell=_CellDcop(i__uA=v_bl_node__V * 2.0),
        label="chunked",
    )


def _fold(results: list[_Dcop], chunks: list[_Chunk], leading: tuple[int, ...]) -> _Dcop:
    """Drive the fold exactly as the chunk loop drives it."""
    fold = _ResultFold(results[0], total=math.prod(leading))
    for result, chunk in zip(results, chunks, strict=True):
        fold.write(result, start=chunk.start, stop=chunk.stop)
    return fold.result(leading_shape=leading)


def test_fold_round_trips_every_field() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    chunks = _chunks(leading, 4)

    results = [_solve(snap, chunk.coords, leading) for chunk in chunks]
    actual = _fold(results, chunks, leading)

    whole_coords = tuple(torch.unravel_index(torch.arange(6), leading))
    expected = _solve(snap, whole_coords, leading)
    assert actual.v_bl_node__V.shape == (2, 3, _COL, _ROW)
    assert actual.i_bl_driver__uA.shape == (2, 3, _COL)
    torch.testing.assert_close(actual.v_bl_node__V.reshape(6, _COL, _ROW), expected.v_bl_node__V)
    torch.testing.assert_close(actual.i_bl_driver__uA.reshape(6, _COL), expected.i_bl_driver__uA)
    torch.testing.assert_close(actual.cell.i__uA.reshape(6, _COL, _ROW), expected.cell.i__uA)
    assert actual.label == "chunked"
    assert type(actual.cell) is _CellDcop


def test_fold_allocates_only_the_target() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    chunks = _chunks(leading, 4)
    results = [_solve(snap, chunk.coords, leading) for chunk in chunks]

    with patch("torch.cat", side_effect=AssertionError("the fold must not concatenate")):
        actual = _fold(results, chunks, leading)

    # One allocation per field, at the full leading and nothing more.
    assert _elements(actual.v_bl_node__V) == 6 * _COL * _ROW
    assert _elements(actual.i_bl_driver__uA) == 6 * _COL
    assert _elements(actual.cell.i__uA) == 6 * _COL * _ROW
    # The chunks themselves are never copied into a staging buffer.
    assert _elements(results[0].v_bl_node__V) == 4 * _COL * _ROW


def test_fold_carries_an_absent_optional_field_through() -> None:
    leading = (2,)
    chunks = _chunks(leading, 1)
    results = [_Result(i_port__uA=torch.full((1, _COL), float(i)), energy__fJ=None) for i, _chunk in enumerate(chunks)]

    fold = _ResultFold(results[0], total=2)
    for result, chunk in zip(results, chunks, strict=True):
        fold.write(result, start=chunk.start, stop=chunk.stop)
    actual = fold.result(leading_shape=leading)

    assert actual.energy__fJ is None
    torch.testing.assert_close(actual.i_port__uA[1], torch.ones(_COL))


def test_fold_preserves_dataclass_replace_semantics() -> None:
    leading = (2,)
    chunks = _chunks(leading, 1)
    results = [
        _Dcop(
            i_bl_driver__uA=torch.full((1, _COL), float(i)),
            v_bl_node__V=torch.full((1, _COL, _ROW), float(i)),
            cell=_CellDcop(i__uA=torch.full((1, _COL, _ROW), float(i))),
            label="chunked",
        )
        for i, _chunk in enumerate(chunks)
    ]

    actual = _fold(results, chunks, leading)

    replaced = replace(actual, label="chunked")
    assert type(replaced) is _Dcop
    assert replaced.i_bl_driver__uA is actual.i_bl_driver__uA
    assert replaced.v_bl_node__V is actual.v_bl_node__V
    assert replaced.cell is actual.cell
    assert replaced.label == "chunked"
    torch.testing.assert_close(actual.i_bl_driver__uA[1], torch.ones(_COL))


# --- The executor over one typed operand tree and one unary function ---


@dataclass(frozen=True)
class _RunOperands:
    cell_snap: _CellSnap
    bl_driver_snap: _ColumnSnap
    v_wl__V: Tensor


class _SpyRun:
    """Record each sliced operand tree and return a deterministic projection."""

    def __init__(self, *, segment_r__MOhm: float) -> None:
        self.segment_r__MOhm = segment_r__MOhm
        self.calls: list[_RunOperands] = []
        self.results: list[_PortResult] = []

    def __call__(self, operands: _RunOperands) -> _PortResult:
        self.calls.append(operands)
        cell_snap = operands.cell_snap
        bl_snap = operands.bl_driver_snap
        # Shape: [..., col, row]
        node = cell_snap.v_wl__V * cell_snap.g__uS + bl_snap.v_ref__V.unsqueeze(-1)
        # Shape: [..., col]
        port = node.sum(dim=-1) + self.segment_r__MOhm
        result = _PortResult(i_port__uA=port, wl_sum__V=operands.v_wl__V.sum(dim=-1))
        self.results.append(result)
        return result


def _cell_snap(leading: tuple[int, ...]) -> _CellSnap:
    """One word line per row held across the columns, over a shared cell buffer."""
    return _CellSnap(
        v_wl__V=torch.randn(*leading, 1, _ROW).expand(*leading, _COL, _ROW),
        g__uS=torch.randn(_COL, _ROW).expand(*leading, _COL, _ROW),
    )


def _clamp_snap(leading: tuple[int, ...]) -> _ColumnSnap:
    return _ColumnSnap(
        v_ref__V=torch.randn(*leading, _COL),
        r_out__MOhm=torch.zeros(()).expand(*leading, _COL),
    )


def _call(
    chunk_size: int,
    leading: tuple[int, ...],
    segment_r: float,
    *,
    run: _SpyRun | None = None,
) -> _PortResult:
    torch.manual_seed(5)
    operands = _RunOperands(
        cell_snap=_cell_snap(leading),
        bl_driver_snap=_clamp_snap(leading),
        v_wl__V=torch.randn(*leading, _ROW),
    )
    return execute_chunked(
        chunk_size=chunk_size,
        leading_shape=leading,
        operands=operands,
        run=run if run is not None else _SpyRun(segment_r__MOhm=segment_r),
    )


def test_operand_tree_is_sliced_before_each_run() -> None:
    segment_r = 1e-4
    run = _SpyRun(segment_r__MOhm=segment_r)
    _call(4, (2, 3), segment_r, run=run)

    assert run.segment_r__MOhm == segment_r
    for operands in run.calls:
        assert type(operands) is _RunOperands
        assert type(operands.cell_snap) is _CellSnap
        assert operands.cell_snap.v_wl__V.shape == (4, _COL, _ROW)
        assert operands.bl_driver_snap.v_ref__V.shape == (4, _COL)
        assert operands.v_wl__V.shape == (4, _ROW)
        # A field that is stride-0 over the whole leading keeps a size-1 axis.
        assert operands.cell_snap.g__uS.shape == (1, _COL, _ROW)


def test_the_caller_states_the_leading_and_the_tail_is_padded() -> None:
    run = _SpyRun(segment_r__MOhm=1e-4)
    _call(4, (2, 3), 1e-4, run=run)

    # 6 leading positions in chunks of 4: two calls, both at the same shape.
    assert len(run.calls) == 2
    assert {operands.cell_snap.v_wl__V.shape for operands in run.calls} == {(4, _COL, _ROW)}


def test_run_size_does_not_exceed_the_complete_leading() -> None:
    run = _SpyRun(segment_r__MOhm=1e-4)
    _call(100, (2, 3), 1e-4, run=run)

    [operands] = run.calls
    assert operands.cell_snap.v_wl__V.shape == (6, _COL, _ROW)
    assert operands.bl_driver_snap.v_ref__V.shape == (6, _COL)
    assert operands.v_wl__V.shape == (6, _ROW)


def test_chunk_coordinates_allocate_one_arange_per_execution() -> None:
    with patch("neurox.primitive.xbar.solver.chunking.torch.arange", wraps=torch.arange) as arange:
        _call(4, (2, 3), 1e-4)

    arange.assert_called_once()


def test_folded_result_equals_one_whole_leading_run() -> None:
    leading = (2, 3)
    segment_r = 1e-4
    chunked = _call(4, leading, segment_r)
    whole = _call(0, leading, segment_r)

    assert type(chunked) is _PortResult
    assert chunked.i_port__uA.shape == (*leading, _COL)
    assert chunked.wl_sum__V.shape == leading
    torch.testing.assert_close(chunked.i_port__uA, whole.i_port__uA, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked.wl_sum__V, whole.wl_sum__V, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("leading_shape", "chunk_size"),
    [
        ((7,), 3),
        ((2, 1, 3), 5),
        ((1, 2, 1, 3), 8),
    ],
)
def test_any_leading_rank_and_chunk_size_match_the_whole_call(
    leading_shape: tuple[int, ...],
    chunk_size: int,
) -> None:
    chunked = _call(chunk_size, leading_shape, 1e-4)
    whole = _call(0, leading_shape, 1e-4)

    torch.testing.assert_close(chunked.i_port__uA, whole.i_port__uA, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked.wl_sum__V, whole.wl_sum__V, rtol=0.0, atol=0.0)


def test_zero_chunk_size_passes_the_full_leading_through() -> None:
    leading = (2, 3)
    operands = _RunOperands(
        cell_snap=_cell_snap(leading),
        bl_driver_snap=_clamp_snap(leading),
        v_wl__V=torch.randn(*leading, _ROW),
    )
    run = _SpyRun(segment_r__MOhm=1e-4)

    result = execute_chunked(
        chunk_size=0,
        leading_shape=leading,
        operands=operands,
        run=run,
    )

    assert run.calls == [operands]
    assert result is run.results[0]


def test_an_operand_short_of_the_declared_leading_is_rejected() -> None:
    operands = _RunOperands(
        cell_snap=_cell_snap((2, 3)),
        # One leading axis short: the slicer would gather its column axis.
        bl_driver_snap=_clamp_snap((3,)),
        v_wl__V=torch.randn(2, 3, _ROW),
    )

    with pytest.raises(ValueError, match="every operand tensor"):
        execute_chunked(
            chunk_size=4,
            leading_shape=(2, 3),
            operands=operands,
            run=_SpyRun(segment_r__MOhm=1e-4),
        )


def test_a_bare_tensor_field_short_of_the_declared_leading_is_rejected() -> None:
    operands = _RunOperands(
        cell_snap=_cell_snap((2, 3)),
        bl_driver_snap=_clamp_snap((2, 3)),
        v_wl__V=torch.randn(3, _ROW),
    )

    with pytest.raises(ValueError, match="every operand tensor"):
        execute_chunked(
            chunk_size=4,
            leading_shape=(2, 3),
            operands=operands,
            run=_SpyRun(segment_r__MOhm=1e-4),
        )


def test_a_call_without_any_leading_reaches_the_run_untouched() -> None:
    run = _SpyRun(segment_r__MOhm=1e-4)
    operands = _RunOperands(
        cell_snap=_cell_snap(()),
        bl_driver_snap=_clamp_snap(()),
        v_wl__V=torch.randn(_ROW),
    )

    result = execute_chunked(chunk_size=4, leading_shape=(), operands=operands, run=run)

    [current] = run.calls
    assert current is operands
    assert current.cell_snap.v_wl__V.shape == (_COL, _ROW)
    assert current.bl_driver_snap.v_ref__V.shape == (_COL,)
    assert current.v_wl__V.shape == (_ROW,)
    assert result.i_port__uA.shape == (_COL,)
    # Returned as it stands: one chunk, nothing allocated and nothing copied.
    assert result is run.results[0]


# --- The layer at work: the fold through a real 1T1R array ---

_ARRAY_COL = 8
_ARRAY_LEADING = 32
_ARRAY_CHUNK = 8
_ARRAY_ROW_SHORT = 16
_ARRAY_ROW_TALL = 128
_VALUE_ROW = 6
_VALUE_LEADING = 7
_BL_V_REF__V = 0.3
_SL_V_REF__V = 0.1
_V_DD_WL__V = 1.1
_V_DD_BL__V = 0.9
_DTYPE = torch.float64
type _Array = XbarArray1t1r[XbarArray1t1rConfig, XbarArray1t1rPolicy]


def _array(*, row_num: int, chunk_size: int) -> _Array:
    """Hand-written tiny 1T1R array with a fully linear cell, every policy off."""
    cell_config = XbarCell1t1rLinearConfig(
        g_cell_off_table__uS=(4.0, 5.0),
        g_cell_on_table__uS=(50.0, 100.0),
        vx_ratio_off_table=(0.5, 0.5),
        vx_ratio_on_table=(0.4, 0.6),
        v_wl_on_threshold__V=0.5,
    )
    array = XbarArray1t1r(
        config=XbarArray1t1rConfig(
            row_cell_space__um=1.0,
            col_cell_space__um=1.0,
            bl_segment_r__MOhm=1e-4,
            sl_segment_r__MOhm=2e-4,
            bl_node_c__fF=0.1,
            x_node_c__fF=0.1,
            sl_node_c__fF=0.1,
            wl_node_c__fF=0.1,
            cell_config=cell_config,
            solver_config=ColBlColSlSolverConfig(n_outer=2, n_inner=2),
        ),
        policy=XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=chunk_size),
        inst_shape=(),
        row_num=row_num,
        col_num=_ARRAY_COL,
        scan_mode=XbarArray1t1rScanMode.WL_IN_BL_SCAN,
        v_dd_wl__V=_V_DD_WL__V,
        v_dd_bl__V=_V_DD_BL__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    # Alternate the two table states so both entries are exercised.
    array.program((torch.arange(_ARRAY_COL * row_num) % 2).reshape(_ARRAY_COL, row_num))
    stamp_names(array)
    return array


def _ideal_driver() -> VoltageDriver:
    """Boundary clamp with `r_out = 0` and every nonideality off."""
    driver = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(_ARRAY_COL,),
        dtype=_DTYPE,
        T__K=300.0,
    )
    driver.eval()
    driver.fabricate()
    stamp_names(driver)
    return driver


def _solve_array(array: _Array, v_wl: Tensor) -> XbarArray1t1rSteadyState:
    """Settle one array against two fresh ideal clamps at fixed references.

    Whoever drives the array bills it: the caller owns the event structure,
    so this helper takes both clamp snaps at the full per-call shape, passes
    the per-row word-line drive through unchanged, and delivers each boundary
    at the converged port state, exactly as a macro does.
    """
    leading = tuple(v_wl.shape[:-1])
    bl_driver = _ideal_driver()
    sl_driver = _ideal_driver()
    bl_ref = torch.full((*leading, _ARRAY_COL), _BL_V_REF__V, dtype=_DTYPE)
    sl_ref = torch.full((*leading, _ARRAY_COL), _SL_V_REF__V, dtype=_DTYPE)
    state = array.solve_array(
        v_wl,
        bl_driver=bl_driver,
        bl_driver_snap=bl_driver.snapshot(v_ref__V=bl_ref, shape=bl_ref.shape),
        sl_driver=sl_driver,
        sl_driver_snap=sl_driver.snapshot(v_ref__V=sl_ref, shape=sl_ref.shape),
    )
    bl_driver.drive(state.i_bl_port__uA, state.v_bl_clamp__V)
    sl_driver.drive(state.i_sl_port__uA, state.v_sl_drive__V)
    return state


def _live_tensor_bytes() -> int:
    """Storage bytes every reachable CPU tensor holds, counted once per pointer.

    The cycle collector runs first so a sample counts reachable state only;
    module trees are cyclic, and an uncollected one would otherwise drift
    between two samples of the same call.
    """
    gc.collect()
    storages: dict[int, int] = {}
    for obj in gc.get_objects():
        # `type(obj)` rather than `isinstance`: reading `__class__` off an
        # unrelated object can trip a library's deprecation shim.
        if not issubclass(type(obj), Tensor):
            continue
        if obj.layout is not torch.strided or obj.device.type != "cpu":
            continue
        try:
            storage = obj.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        except RuntimeError:
            # A tracing-time tensor (FakeTensor, FunctionalTensor) has no
            # storage to address and holds no memory to count. Compiled
            # artifacts of earlier tests keep such tensors reachable.
            continue
    return sum(storages.values())


@contextmanager
def _chunk_boundary_bytes() -> Iterator[list[int]]:
    """Sample the live tensor bytes at every chunk boundary of the enclosed solve.

    The chunk generator resumes only once the previous chunk's body has run
    to completion, so a sample sees exactly what the loop carries between
    chunks: the preallocated fold plus the chunk coordinates, and any
    per-chunk state that failed to die with its chunk.
    """
    samples: list[int] = []
    unpatched = _iter_chunks

    def probing(**kwargs: Any) -> Iterator[_Chunk]:
        for chunk in unpatched(**kwargs):
            samples.append(_live_tensor_bytes())
            yield chunk

    with patch("neurox.primitive.xbar.solver.chunking._iter_chunks", probing):
        yield samples


def _retained_bytes(*, row_num: int) -> int:
    """Bytes the chunk loop carries across a boundary, over its own entry state.

    The solve runs eagerly. The law is about the chunk loop, which is a
    declared eager island, while the tall geometry the law needs would take the `dynamic=False`
    solver body minutes to unroll; a traced body also hands the probe
    storage-less tensors, which carry no bytes to count.
    """
    array = _array(row_num=row_num, chunk_size=_ARRAY_CHUNK)
    v_wl = torch.rand(_ARRAY_LEADING, row_num, dtype=_DTYPE) * 1.2
    with (
        torch._dynamo.config.patch(disable=True),
        _chunk_boundary_bytes() as samples,
        Profiler(leading_rank=1),
    ):
        _solve_array(array, v_wl)
    return max(samples) - samples[0]


def test_chunk_size_moves_neither_the_port_state_nor_the_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk size is a memory knob: every chunking of one call agrees bit for bit."""
    v_wl = torch.rand(_VALUE_LEADING, _VALUE_ROW, dtype=_DTYPE) * 1.2
    folded: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    for chunk_size in (0, 2, 3, 5, 100):
        array = _array(row_num=_VALUE_ROW, chunk_size=chunk_size)
        billed: list[Tensor] = []
        monkeypatch.setattr(array, "_record_dynamic_energy", billed.append)
        with Profiler(leading_rank=1):
            state = _solve_array(array, v_wl)
        [energy__fJ] = billed
        folded[chunk_size] = (state.i_bl_port__uA, state.v_bl_clamp__V, energy__fJ)

    # The energy is billed once, at the full leading, after the fold.
    assert folded[0][2].shape == (_VALUE_LEADING,)
    # `0` is the un-chunked solve: one projection over the whole leading.
    whole = folded[0]
    for chunk_size, (i_bl_port__uA, v_bl_clamp__V, energy__fJ) in folded.items():
        assert torch.equal(i_bl_port__uA, whole[0]), chunk_size
        assert torch.equal(v_bl_clamp__V, whole[1]), chunk_size
        assert torch.equal(energy__fJ, whole[2]), chunk_size


def test_retained_state_is_flat_in_row_num() -> None:
    """What survives a chunk is per-column state, so a taller array retains no more."""
    short = _retained_bytes(row_num=_ARRAY_ROW_SHORT)
    tall = _retained_bytes(row_num=_ARRAY_ROW_TALL)

    # Four port fields plus the energy, all at [leading, col] or [leading].
    port_bytes = (4 * _ARRAY_COL + 1) * _ARRAY_LEADING * _DTYPE.itemsize
    grid_bytes = _ARRAY_COL * _ARRAY_ROW_TALL * _ARRAY_LEADING * _DTYPE.itemsize

    # Flat in row_num: an eight-fold taller array carries the same state.
    assert abs(tall - short) <= 1024
    # And that state is the port state itself, far below a single grid field.
    assert tall <= 2 * port_bytes
    assert tall < grid_bytes // 8
