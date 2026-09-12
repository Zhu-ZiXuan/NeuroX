"""Tests for shared balanced chunk execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common import chunking as chunking_module
from neurox.common.chunking import run_chunked
from neurox.common.pytree_dataclass_mixin import PyTreeDataClassMixin
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin

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


def _elements(tensor: Tensor) -> int:
    return tensor.untyped_storage().size() // tensor.element_size()


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


def test_slices_expose_exact_shapes_without_materializing_broadcast_storage(device: torch.device) -> None:
    leading_shape = (2, 3)
    leading_size = 6
    dense = torch.arange(leading_size * _COL * _ROW, device=device).reshape(*leading_shape, _COL, _ROW)
    shared = torch.arange(_COL * _ROW, device=device).reshape(_COL, _ROW).expand_as(dense)
    partial = torch.arange(3 * _ROW, device=device).reshape(1, 3, 1, _ROW).expand_as(dense)

    class BroadcastOperands(TensorDataClassMixin, PyTreeDataClassMixin):
        dense: Tensor
        shared: Tensor
        partial: Tensor

    flat_indices = torch.arange(3, device=device)
    coords = tuple(torch.unravel_index(flat_indices, leading_shape))
    slice_operands = chunking_module._slice_operands
    current = slice_operands(
        BroadcastOperands(dense=dense, shared=shared, partial=partial),
        coords=coords,
        chunk_size=3,
        leading_shape=leading_shape,
    )

    assert current.dense.shape == current.shared.shape == current.partial.shape == (3, _COL, _ROW)
    assert _elements(current.dense) == 3 * _COL * _ROW
    assert _elements(current.shared) == _COL * _ROW
    assert _elements(current.partial) == 3 * _ROW
    torch.testing.assert_close(current.dense, dense.reshape(leading_size, _COL, _ROW)[:3])
    torch.testing.assert_close(current.shared, shared.reshape(leading_size, _COL, _ROW)[:3])
    torch.testing.assert_close(current.partial, partial.reshape(leading_size, _COL, _ROW)[:3])


@pytest.mark.parametrize(
    ("leading_shape", "expected_chunk_size", "flat_size"),
    [
        ((2, 3), 0, 6),
        ((1, 1), 4, 1),
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
            values=current.values.square(),
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
    torch.testing.assert_close(result.values, operands.values.square())
    torch.testing.assert_close(result.leaf.doubled, operands.leaf.grid * 2)


def test_empty_leading_shape_acquires_only_the_body_chunk_axis(device: torch.device) -> None:
    operands = _operands((), device=device)

    def body_fn(current: _Operands) -> _Result:
        return _Result(
            values=current.values + 1,
            leaf=_ResultLeaf(doubled=current.leaf.grid * 2),
            absent=None,
        )

    result = run_chunked(
        expected_chunk_size=0,
        leading_shape=(),
        device=device,
        operands=operands,
        output_template=_output_template(device=device),
        body_fn=body_fn,
    )

    assert result.values.shape == torch.Size()
    assert result.leaf.doubled.shape == (_COL, _ROW)
    torch.testing.assert_close(result.values, operands.values + 1)
    torch.testing.assert_close(result.leaf.doubled, operands.leaf.grid * 2)


@pytest.mark.parametrize(
    ("leading_shape", "expected_chunk_size", "message"),
    [
        ((2, 3), -1, "expected_chunk_size must be nonnegative"),
        ((0, 3), 4, "leading_shape must contain at least one position"),
    ],
)
def test_rejects_invalid_schedules_before_calling_the_body(
    leading_shape: tuple[int, ...],
    expected_chunk_size: int,
    message: str,
    device: torch.device,
) -> None:
    def body_fn(_current: _Operands) -> _Result:
        raise AssertionError("body called")

    with pytest.raises(ValueError, match=message):
        run_chunked(
            expected_chunk_size=expected_chunk_size,
            leading_shape=leading_shape,
            device=device,
            operands=_operands(leading_shape, device=device),
            output_template=_output_template(device=device),
            body_fn=body_fn,
        )


def test_chunk_bound_larger_than_work_uses_one_complete_call(device: torch.device) -> None:
    leading_shape = (2, 3)
    operands = _operands(leading_shape, device=device)

    def body_fn(current: _Operands) -> _Result:
        return _Result(
            values=current.values,
            leaf=_ResultLeaf(doubled=current.leaf.grid),
            absent=None,
        )

    result = run_chunked(
        expected_chunk_size=100,
        leading_shape=leading_shape,
        device=device,
        operands=operands,
        output_template=_output_template(device=device),
        body_fn=body_fn,
    )

    torch.testing.assert_close(result.values, operands.values)


@pytest.mark.parametrize(("leading_size", "chunk_size", "scan_count"), [(6, 3, 1), (9, 3, 1), (13, 3, 2), (1, 0, 0)])
def test_scan_collects_only_chunk_outputs(
    leading_size: int, chunk_size: int, scan_count: int, device: torch.device
) -> None:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph: torch.fx.GraphModule, _inputs: list[object]):
        graphs.append(graph)
        return graph.forward

    def body_fn(current: _Operands) -> _Result:
        return _Result(
            values=current.values.square(),
            leaf=_ResultLeaf(doubled=current.leaf.grid * 2),
            absent=None,
        )

    def run():
        return run_chunked(
            expected_chunk_size=chunk_size,
            leading_shape=(leading_size,),
            device=device,
            operands=_operands((leading_size,), device=device),
            output_template=_output_template(device=device),
            body_fn=body_fn,
        )

    result = torch.compile(run, backend=backend, fullgraph=True)()
    torch.testing.assert_close(result.values, torch.arange(leading_size, device=device).square())
    assert len(graphs) == 1
    nodes = [
        node
        for module in graphs[0].modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    ]
    scans = [node for node in nodes if node.target is torch.ops.higher_order.scan]
    assert len(scans) == scan_count
    assert all(node.target is not torch.ops.higher_order.while_loop for node in nodes)
    assert all("index_copy" not in str(node.target) for node in nodes)
    for node in scans:
        # Native scan carries only the hidden placeholder; all business data are outputs.
        assert len(node.args[1]) == 1
        assert node.args[1][0].meta["example_value"].numel() == 0
    assert _elements(result.values) == leading_size
    assert _elements(result.leaf.doubled) == leading_size * _COL * _ROW


@pytest.mark.parametrize("fullgraph", [False, True])
def test_caller_owns_scalar_capture_configuration(device: torch.device, fullgraph: bool) -> None:
    def body_fn(current: _Operands) -> _Result:
        return _Result(
            values=current.values.square(),
            leaf=_ResultLeaf(doubled=current.leaf.grid * 2),
            absent=None,
        )

    def run(operands: _Operands) -> _Result:
        result = run_chunked(
            expected_chunk_size=2,
            leading_shape=(6,),
            device=device,
            operands=operands,
            output_template=_output_template(device=device),
            body_fn=body_fn,
        )
        return result

    capture_scalar_outputs = torch._dynamo.config.capture_scalar_outputs
    operands = _operands((6,), device=device)
    # Scalar capture is configured by the caller before its compile boundary.
    with torch._dynamo.config.patch(capture_scalar_outputs=not fullgraph):
        result = torch.compile(run, fullgraph=fullgraph)(operands)
    torch.testing.assert_close(result.values, operands.values.square())
    assert torch._dynamo.config.capture_scalar_outputs == capture_scalar_outputs


def test_scan_cuda_peak_avoids_whole_result_copies(device: torch.device) -> None:
    if device.type != "cuda":
        pytest.skip("CUDA allocation statistics require a CUDA device")

    leading_size = 512
    width = 32768
    row = torch.linspace(0, 1, width, device=device)
    operands = _Leaf(grid=row.expand(leading_size, width))

    def body_fn(current: _Leaf) -> _Result:
        return _Result(
            values=current.grid.square() + 1,
            leaf=_ResultLeaf(doubled=current.grid.sum(-1)),
            absent=None,
        )

    def run() -> _Result:
        result = run_chunked(
            expected_chunk_size=32,
            leading_shape=(leading_size,),
            device=device,
            operands=operands,
            output_template=_output_template(device=device),
            body_fn=body_fn,
        )
        return result

    compiled_run = torch.compile(run, dynamic=False, fullgraph=True)
    with torch.no_grad():
        warmup = compiled_run()
        torch.cuda.synchronize(device)
        del warmup
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = compiled_run()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline

    output_bytes = (leading_size * width + leading_size) * row.element_size()
    # One uniform scan needs its output and chunk workspace, not two full results.
    assert peak < 2 * output_bytes, (peak, output_bytes)
    torch.testing.assert_close(result.values, (row.square() + 1).expand(leading_size, width))


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
