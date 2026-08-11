"""Tests for fixed-shape solver chunk construction, snap slicing, and folding.

Covers the chunk coordinates and tail padding, the collapse-index-re-expand
slicing rule (asserted on storage size, not only values), the leading-shape
broadcast helper, and the preallocating fold that writes each chunk's
measurement into its global positions.

The last section drives the whole layer through a real 1T1R array and pins
the two laws the fold exists for:

  * chunk size is a memory knob — the port state and the billed energy are
    bit-identical across chunk sizes, the un-chunked solve included,
  * what survives a chunk scales with the columns, not with the cells: the
    state retained across chunk boundaries is flat in ``row_num``.

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

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rOperationMode,
    XbarArray1t1rPolicy,
    XbarArray1t1rSteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy
from neurox.primitive.xbar.solver import (
    ChunkSpec,
    MeasureFold,
    NestedParallelRailSolverConfig,
    iter_chunks,
    slice_snap,
    slice_tensor,
)

_COL = 4
_ROW = 5
_CPU = torch.device("cpu")


@dataclass(frozen=True)
class _DeviceSnap:
    """Nested device snap on the same cell grid."""

    g__uS: Tensor


@dataclass(frozen=True)
class _GridSnap:
    """Cell-grid snap holding one field per row of the broadcast table."""

    buffer: Tensor
    wl_drive: Tensor
    partial: Tensor
    full: Tensor
    device: _DeviceSnap
    t_elapsed: float


@dataclass(frozen=True)
class _ColumnSnap:
    """Per-column clamp snap with a constant field expanded over the call."""

    v_ref__V: Tensor
    r_out__MOhm: Tensor


@dataclass(frozen=True)
class _CellDcop:
    """Nested per-chunk cell result."""

    i__uA: Tensor


@dataclass(frozen=True)
class _Dcop:
    """Per-chunk solver result with one nested dataclass field."""

    i_bl_driver: Tensor
    v_bl_node: Tensor
    cell: _CellDcop
    label: str


@dataclass(frozen=True)
class _Measure:
    """Per-chunk measurement carrying one optional field."""

    i_port__uA: Tensor
    energy__fJ: Tensor | None


def _elements(t: Tensor) -> int:
    """Storage elements a tensor actually occupies."""
    return t.untyped_storage().size() // t.element_size()


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
    specs = list(iter_chunks(leading=(5,), chunk_size=3, device=_CPU))

    assert [spec.solve_size for spec in specs] == [3, 3]
    assert [spec.valid_size for spec in specs] == [3, 2]
    torch.testing.assert_close(specs[0].multi_coords[0], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(specs[1].multi_coords[0], torch.tensor([3, 4, 4]))
    torch.testing.assert_close(specs[1].flat_global_idx, torch.tensor([3, 4]))


def test_non_positive_chunk_size_keeps_one_unpadded_chunk() -> None:
    [spec] = list(iter_chunks(leading=(2, 3), chunk_size=0, device=_CPU))

    assert spec.solve_size == 6
    assert spec.valid_size == 6
    torch.testing.assert_close(spec.flat_global_idx, torch.arange(6))


def test_chunk_larger_than_workload_still_uses_fixed_solver_size() -> None:
    [spec] = list(iter_chunks(leading=(2,), chunk_size=5, device=_CPU))

    assert spec.solve_size == 5
    assert spec.valid_size == 2
    torch.testing.assert_close(spec.multi_coords[0], torch.tensor([0, 1, 1, 1, 1]))


def test_empty_leading_shape_remains_atomic_and_unpadded() -> None:
    [spec] = list(iter_chunks(leading=(), chunk_size=5, device=_CPU))

    assert spec.multi_coords == ()
    assert spec.solve_size == 1
    assert spec.valid_size == 1


def test_slicer_storage_follows_the_broadcast_table() -> None:
    leading = (2, 3)
    chunk_size = 4
    snap = _grid_snap(leading)
    [spec, _tail] = list(iter_chunks(leading=leading, chunk_size=chunk_size, device=_CPU))

    sliced = slice_snap(snap, coords=spec.multi_coords, leading=leading)

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
    [spec, _tail] = list(iter_chunks(leading=leading, chunk_size=chunk_size, device=_CPU))

    sliced = slice_snap(snap, coords=spec.multi_coords, leading=leading)

    assert sliced.g__uS.shape == (chunk_size, _COL, _ROW)
    assert _elements(sliced.g__uS) == chunk_size * _ROW
    assert _elements(wl_drive[spec.multi_coords]) == chunk_size * _COL * _ROW
    torch.testing.assert_close(sliced.g__uS, wl_drive[spec.multi_coords])


def test_slicer_values_match_a_dense_gather() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    [spec, tail] = list(iter_chunks(leading=leading, chunk_size=4, device=_CPU))

    for chunk_spec in (spec, tail):
        coords = chunk_spec.multi_coords
        sliced = slice_snap(snap, coords=coords, leading=leading)

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
    [spec, _tail] = list(iter_chunks(leading=leading, chunk_size=4, device=_CPU))

    sliced = slice_snap(snap, coords=spec.multi_coords, leading=leading)

    torch.testing.assert_close(sliced.device.g__uS, snap.device.g__uS[spec.multi_coords])
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
    [spec, _tail] = list(iter_chunks(leading=leading, chunk_size=chunk_size, device=_CPU))

    sliced = slice_snap(snap, coords=spec.multi_coords, leading=leading)

    # One trailing axis behind the leading, and it is the one that survives.
    assert sliced.v_ref__V.shape == (chunk_size, _COL)
    assert _elements(sliced.v_ref__V) == chunk_size * _COL
    # A field that is stride-0 throughout costs one element and broadcasts back.
    assert sliced.r_out__MOhm.shape == (1, _COL)
    assert _elements(sliced.r_out__MOhm) == 1


def test_slicer_keeps_the_chunk_axis_on_a_singleton_leading() -> None:
    leading = (1,)
    snap = _ColumnSnap(v_ref__V=torch.randn(1, _COL), r_out__MOhm=torch.zeros(()).expand(1, _COL))
    [spec] = list(iter_chunks(leading=leading, chunk_size=3, device=_CPU))

    sliced = slice_snap(snap, coords=spec.multi_coords, leading=leading)

    assert sliced.v_ref__V.shape == (3, _COL)


def test_slicer_passes_through_when_there_is_no_leading() -> None:
    snap = _ColumnSnap(v_ref__V=torch.randn(_COL), r_out__MOhm=torch.zeros(()))
    [spec] = list(iter_chunks(leading=(), chunk_size=3, device=_CPU))

    assert slice_snap(snap, coords=spec.multi_coords, leading=()) is snap


def test_slicer_rejects_a_field_of_lower_rank_than_the_leading() -> None:
    snap = _ColumnSnap(v_ref__V=torch.zeros(2, _COL), r_out__MOhm=torch.zeros(()))
    [spec] = list(iter_chunks(leading=(2,), chunk_size=2, device=_CPU))

    with pytest.raises(ValueError, match="rank >= len"):
        slice_snap(snap, coords=spec.multi_coords, leading=(2,))


def test_a_bare_tensor_slices_by_the_same_rule_as_a_snap_field() -> None:
    """The dataclass is a walk, not a precondition: the rule is the tensor's."""
    leading = (2, 3)
    chunk_size = 4
    wl_drive = torch.randn(*leading, 1, _ROW).expand(*leading, _COL, _ROW)
    [spec, _tail] = list(iter_chunks(leading=leading, chunk_size=chunk_size, device=_CPU))

    bare = slice_tensor(wl_drive, coords=spec.multi_coords, leading=leading)
    wrapped = slice_snap(_DeviceSnap(g__uS=wl_drive), coords=spec.multi_coords, leading=leading)

    # Same shape, same values, and the same collapsed storage.
    assert bare.shape == (chunk_size, _COL, _ROW)
    assert _elements(bare) == chunk_size * _ROW
    torch.testing.assert_close(bare, wrapped.g__uS)
    assert _elements(bare) == _elements(wrapped.g__uS)


def test_a_bare_tensor_passes_through_when_there_is_no_leading() -> None:
    v_wl = torch.randn(_ROW)
    [spec] = list(iter_chunks(leading=(), chunk_size=3, device=_CPU))

    assert slice_tensor(v_wl, coords=spec.multi_coords, leading=()) is v_wl


def _solve(snap: _GridSnap, coords: tuple[Tensor, ...], leading: tuple[int, ...]) -> _Dcop:
    """Stand-in solver: one fixed-shape pass over the chunk."""
    sliced = slice_snap(snap, coords=coords, leading=leading)
    v_bl_node = sliced.buffer + sliced.wl_drive + sliced.partial + sliced.full
    return _Dcop(
        i_bl_driver=v_bl_node.sum(dim=-1),
        v_bl_node=v_bl_node,
        cell=_CellDcop(i__uA=v_bl_node * 2.0),
        label="chunked",
    )


def _fold(chunks: list[_Dcop], specs: list[ChunkSpec], leading: tuple[int, ...]) -> _Dcop:
    """Drive the fold exactly as the chunk loop drives it."""
    fold = MeasureFold(chunks[0], b_total=math.prod(leading))
    for chunk, spec in zip(chunks, specs, strict=True):
        fold.write(chunk, flat_idx=spec.flat_global_idx)
    return fold.result(leading=leading)


def test_fold_round_trips_every_field() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    specs = list(iter_chunks(leading=leading, chunk_size=4, device=_CPU))

    chunks = [_solve(snap, spec.multi_coords, leading) for spec in specs]
    actual = _fold(chunks, specs, leading)

    whole_coords = tuple(torch.unravel_index(torch.arange(6), leading))
    expected = _solve(snap, whole_coords, leading)
    assert actual.v_bl_node.shape == (2, 3, _COL, _ROW)
    assert actual.i_bl_driver.shape == (2, 3, _COL)
    torch.testing.assert_close(actual.v_bl_node.reshape(6, _COL, _ROW), expected.v_bl_node)
    torch.testing.assert_close(actual.i_bl_driver.reshape(6, _COL), expected.i_bl_driver)
    torch.testing.assert_close(actual.cell.i__uA.reshape(6, _COL, _ROW), expected.cell.i__uA)
    assert actual.label == "chunked"
    assert type(actual.cell) is _CellDcop


def test_fold_allocates_only_the_target() -> None:
    leading = (2, 3)
    snap = _grid_snap(leading)
    specs = list(iter_chunks(leading=leading, chunk_size=4, device=_CPU))
    chunks = [_solve(snap, spec.multi_coords, leading) for spec in specs]

    with patch("torch.cat", side_effect=AssertionError("the fold must not concatenate")):
        actual = _fold(chunks, specs, leading)

    # One allocation per field, at the full leading and nothing more.
    assert _elements(actual.v_bl_node) == 6 * _COL * _ROW
    assert _elements(actual.i_bl_driver) == 6 * _COL
    assert _elements(actual.cell.i__uA) == 6 * _COL * _ROW
    # The chunks themselves are never copied into a staging buffer.
    assert _elements(chunks[0].v_bl_node) == 4 * _COL * _ROW


def test_fold_carries_an_absent_optional_field_through() -> None:
    leading = (2,)
    specs = list(iter_chunks(leading=leading, chunk_size=1, device=_CPU))
    chunks = [_Measure(i_port__uA=torch.full((1, _COL), float(i)), energy__fJ=None) for i, _spec in enumerate(specs)]

    fold = MeasureFold(chunks[0], b_total=2)
    for chunk, spec in zip(chunks, specs, strict=True):
        fold.write(chunk, flat_idx=spec.flat_global_idx)
    actual = fold.result(leading=leading)

    assert actual.energy__fJ is None
    torch.testing.assert_close(actual.i_port__uA[1], torch.ones(_COL))


def test_fold_preserves_dataclass_replace_semantics() -> None:
    leading = (2,)
    specs = list(iter_chunks(leading=leading, chunk_size=1, device=_CPU))
    chunks = [
        _Dcop(
            i_bl_driver=torch.full((1, _COL), float(i)),
            v_bl_node=torch.full((1, _COL, _ROW), float(i)),
            cell=_CellDcop(i__uA=torch.full((1, _COL, _ROW), float(i))),
            label="chunked",
        )
        for i, _spec in enumerate(specs)
    ]

    actual = _fold(chunks, specs, leading)

    assert actual == replace(actual, label="chunked")
    torch.testing.assert_close(actual.i_bl_driver[1], torch.ones(_COL))


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


def _array(*, row_num: int, chunk_size: int) -> XbarArray1t1r:
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
            solver_config=NestedParallelRailSolverConfig(n_outer=2, n_inner=2),
        ),
        policy=XbarArray1t1rPolicy(cell_policy=XbarCell1t1rLinearPolicy(), solve_chunk_size=chunk_size),
        inst_shape=(),
        row_num=row_num,
        col_num=_ARRAY_COL,
        operation_mode=XbarArray1t1rOperationMode.WL_IN_BL_SCAN,
        v_dd_wl__V=_V_DD_WL__V,
        v_dd_bl__V=_V_DD_BL__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    # Alternate the two table states so both entries are exercised.
    array.program((torch.arange(_ARRAY_COL * row_num) % 2).reshape(_ARRAY_COL, row_num))
    return array


def _ideal_driver() -> VoltageDriver:
    """Boundary clamp with ``r_out = 0`` and every nonideality off."""
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
    return driver


def _solve_array(array: XbarArray1t1r, v_wl: Tensor) -> XbarArray1t1rSteadyState:
    """Settle one array against two fresh ideal clamps at fixed references.

    Whoever drives the array bills it: the caller owns the event structure,
    so this helper takes both clamp snaps at the full per-call shape, lifts
    the per-row word-line drive onto the cell grid, and delivers each
    boundary at the converged port state, exactly as a macro does.
    """
    leading = tuple(v_wl.shape[:-1])
    row_num = v_wl.shape[-1]
    bl_driver = _ideal_driver()
    sl_driver = _ideal_driver()
    bl_ref = torch.full((*leading, _ARRAY_COL), _BL_V_REF__V, dtype=_DTYPE)
    sl_ref = torch.full((*leading, _ARRAY_COL), _SL_V_REF__V, dtype=_DTYPE)
    # One word line per row, held across every column.
    # Shape: [*leading, row_num] -> [*leading, col_num, row_num]
    v_wl_grid = v_wl.unsqueeze(-2).expand(*leading, _ARRAY_COL, row_num)
    state = array.solve_array(
        v_wl_grid,
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
    unpatched = iter_chunks

    def probing(**kwargs: Any) -> Iterator[ChunkSpec]:
        for spec in unpatched(**kwargs):
            samples.append(_live_tensor_bytes())
            yield spec

    with patch("neurox.primitive.xbar.solver.chunked.iter_chunks", probing):
        yield samples


def _retained_bytes(*, row_num: int) -> int:
    """Bytes the chunk loop carries across a boundary, over its own entry state.

    The solve runs eagerly. The law is about the chunk loop, which is eager
    by construction (``ChunkedSolver.solve_dc`` is ``torch.compiler.disable``d),
    while the tall geometry the law needs would take the ``dynamic=False``
    solver body minutes to unroll; a traced body also hands the probe
    storage-less tensors, which carry no bytes to count.
    """
    array = _array(row_num=row_num, chunk_size=_ARRAY_CHUNK)
    v_wl = torch.rand(_ARRAY_LEADING, row_num, dtype=_DTYPE) * 1.2
    with (
        torch._dynamo.config.patch(disable=True),
        _chunk_boundary_bytes() as samples,
        NeuroxProfiler(leading_rank=1),
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
        with NeuroxProfiler(leading_rank=1):
            state = _solve_array(array, v_wl)
        [energy__fJ] = billed
        folded[chunk_size] = (state.i_bl_port__uA, state.v_bl_clamp__V, energy__fJ)

    # The energy is billed once, at the full leading, after the fold.
    assert folded[0][2].shape == (_VALUE_LEADING,)
    # `0` is the un-chunked solve: one measurement over the whole leading.
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
