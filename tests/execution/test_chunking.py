"""Tests for shared balanced chunk execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin
from neurox.execution.chunking import run_chunked

_COL = 4
_ROW = 5


class _Leaf(TensorDataClassMixin, PyTreeDataClassMixin):
    grid: Tensor


@dataclass(eq=False, frozen=True, kw_only=True)
class _Operands:
    values: Tensor
    leaf: _Leaf
    label: str


class _ResultLeaf(TensorDataClassMixin, PyTreeDataClassMixin):
    doubled: Tensor


@dataclass(eq=False, frozen=True, kw_only=True)
class _Result:
    values: Tensor
    leaf: _ResultLeaf
    absent: Tensor | None


torch.export.register_dataclass(_Result)


def _operands(leading_shape: tuple[int, ...], *, device: torch.device) -> _Operands:
    leading_size = torch.Size(leading_shape).numel()
    values = torch.arange(leading_size, device=device).reshape(leading_shape)
    grid = torch.arange(_COL * _ROW, device=device).reshape(_COL, _ROW)
    return _Operands(
        values=values,
        leaf=_Leaf(grid=grid.expand(*leading_shape, _COL, _ROW)),
        label="kept",
    )


def _output_template(*, device: torch.device) -> _Result:
    return _Result(
        values=torch.empty(0, dtype=torch.int64, device=device),
        leaf=_ResultLeaf(doubled=torch.empty(0, dtype=torch.int64, device=device)),
        absent=None,
    )


@pytest.mark.parametrize("chunk_size", [0, 4])
def test_chunked_execution_preserves_dense_and_broadcast_inputs(device: torch.device, chunk_size: int) -> None:
    leading_shape = (2, 3)
    leading_size = 6
    dense = torch.arange(leading_size * _COL * _ROW, device=device).reshape(*leading_shape, _COL, _ROW)
    shared = torch.arange(_COL * _ROW, device=device).reshape(_COL, _ROW).expand_as(dense)
    partial = torch.arange(3 * _ROW, device=device).reshape(1, 3, 1, _ROW).expand_as(dense)

    class BroadcastOperands(TensorDataClassMixin, PyTreeDataClassMixin):
        dense: Tensor
        shared: Tensor
        partial: Tensor

    operands = BroadcastOperands(dense=dense, shared=shared, partial=partial)

    def body(current: BroadcastOperands) -> BroadcastOperands:
        return BroadcastOperands(dense=current.dense * 2, shared=current.shared * 2, partial=current.partial * 2)

    result = run_chunked(
        expected_chunk_size=chunk_size,
        leading_shape=leading_shape,
        device=device,
        operands=operands,
        output_template=operands,
        body_fn=body,
    )

    torch.testing.assert_close(result.dense, dense * 2)
    torch.testing.assert_close(result.shared, shared * 2)
    torch.testing.assert_close(result.partial, partial * 2)


def test_compiled_chunked_gathers_follow_each_leading_shape(device: torch.device) -> None:
    def body(current: _Operands) -> _Result:
        return _Result(
            values=current.values * 3,
            leaf=_ResultLeaf(doubled=current.values.unsqueeze(-1).unsqueeze(-1) * 7 + current.leaf.grid),
            absent=None,
        )

    @torch.compile(fullgraph=True, dynamic=False)
    def evaluate(values: Tensor, grid: Tensor) -> _Result:
        return run_chunked(
            expected_chunk_size=3,
            leading_shape=tuple(values.shape),
            device=values.device,
            operands=_Operands(values=values, leaf=_Leaf(grid=grid), label="kept"),
            output_template=_output_template(device=values.device),
            body_fn=body,
        )

    # Ten positions yield two mapped groups, each with two chunks: 3, 3, 2, 2.
    # Reverse the axes before revisiting a shape with new data in the same process.
    for sample_index, (row_num, col_num) in enumerate(((2, 5), (5, 2), (2, 5))):
        rows = torch.arange(row_num, device=device).reshape(row_num, 1)
        columns = torch.arange(col_num, device=device).reshape(1, col_num)
        tail = torch.arange(3, device=device).reshape(1, 1, 1, 3)
        shift = 1000 * sample_index
        values = 10 * rows + columns + shift
        grid = (100 * rows.unsqueeze(-1).unsqueeze(-1) + tail).expand(row_num, col_num, 2, 3)

        result = evaluate(values, grid)

        # The oracle uses axis coordinates directly, without chunking or gathers.
        expected_values = 30 * rows + 3 * columns + 3 * shift
        expected_grid = (170 * rows + 7 * columns + 7 * shift).unsqueeze(-1).unsqueeze(-1) + tail
        expected_grid = expected_grid.expand(row_num, col_num, 2, 3)
        assert result.values.shape == (row_num, col_num)
        assert result.leaf.doubled.shape == (row_num, col_num, 2, 3)
        assert result.values.device == result.leaf.doubled.device == values.device
        torch.testing.assert_close(result.values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(result.leaf.doubled, expected_grid, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("leading_shape", "expected_chunk_size", "flat_size"),
    [
        ((2, 3), 0, 6),
        ((2, 3), 100, 6),
        ((1, 1), 4, 1),
        ((), 0, 1),
        ((), 4, 1),
    ],
)
def test_unpartitioned_calls_still_flatten_and_restore(
    leading_shape: tuple[int, ...],
    expected_chunk_size: int,
    flat_size: int,
    device: torch.device,
) -> None:
    operands = _operands(leading_shape, device=device)

    def body_fn(current: _Operands) -> _Result:
        return _Result(
            values=current.values.cumsum(dim=0),
            leaf=_ResultLeaf(doubled=current.leaf.grid * 2),
            absent=None,
        )

    result = run_chunked(
        expected_chunk_size=expected_chunk_size,
        leading_shape=leading_shape,
        device=device,
        operands=operands,
        output_template=_output_template(device=device),
        body_fn=body_fn,
    )

    assert result.values.numel() == flat_size
    assert result.values.shape == leading_shape
    assert result.leaf.doubled.shape == (*leading_shape, _COL, _ROW)
    expected = operands.values.flatten().cumsum(dim=0).reshape(leading_shape)
    torch.testing.assert_close(result.values, expected)
    torch.testing.assert_close(result.leaf.doubled, operands.leaf.grid * 2)


class _TraceResult(TensorDataClassMixin, PyTreeDataClassMixin):
    result: _Result
    trace: _ResultLeaf


@pytest.mark.parametrize(("leading_shape", "chunk_size"), [((2, 5), 3), ((2, 5), 4), ((7,), 3), ((3,), 2), ((), 0)])
def test_raw_histories_share_positional_reassembly(
    device: torch.device, leading_shape: tuple[int, ...], chunk_size: int
) -> None:
    operands = _operands(leading_shape, device=device)

    def body(current: _Operands) -> _TraceResult:
        observation = current.values[..., None, None] + torch.arange(6, device=device).reshape(2, 3)
        trace = torch.where(current.values[..., None, None] % 2 == 0, observation.to(torch.float64), torch.nan)
        return _TraceResult(
            result=_Result(
                values=current.values.square(), leaf=_ResultLeaf(doubled=current.leaf.grid * 2), absent=None
            ),
            trace=_ResultLeaf(doubled=trace),
        )

    result = run_chunked(
        expected_chunk_size=chunk_size,
        leading_shape=leading_shape,
        device=device,
        operands=operands,
        output_template=_TraceResult(
            result=_output_template(device=device), trace=_ResultLeaf(doubled=torch.empty(0, device=device))
        ),
        body_fn=body,
    )
    expected = body(operands)
    torch.testing.assert_close(result.result.values, expected.result.values)
    assert result.trace.doubled.shape == (*leading_shape, 2, 3)
    torch.testing.assert_close(result.trace.doubled, expected.trace.doubled, equal_nan=True)
