"""Shared numerical steps, convergence masks, and scan history storage."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.solving import SolvingState, SolvingTrace, run_solving_loop, run_solving_trace_scan
from neurox.common.torch_compat import torch_assert_async


class _State(SolvingState):
    value: Tensor


class _Trace(SolvingTrace):
    value: Tensor


class _NestedTrace(SolvingTrace):
    value: Tensor
    inner: _Trace


def _count_down(value: Tensor, enabled: Tensor, max_iter: int, record_trace: bool, *, strict: bool = True):
    initial = _State(value=value, is_active=enabled & (value > 0))

    def evaluate(current: _State) -> tuple[_State, _Trace]:
        next_value = torch.where(current.is_active, current.value - 1, current.value)
        return (
            _State(value=next_value, is_active=current.is_active & (next_value > 0)),
            _Trace(value=next_value),
        )

    if record_trace:
        return run_solving_trace_scan(
            init_state=initial,
            body_fn=evaluate,
            default_trace=_Trace(value=torch.full_like(value, torch.nan)),
            max_iter=max_iter,
            strict=strict,
        )

    def solve_body(current: _State) -> _State:
        next_state, _ = evaluate(current)
        return next_state

    return run_solving_loop(init_state=initial, body_fn=solve_body, max_iter=max_iter, strict=strict), None


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("record_trace", [False, True])
def test_shared_step_preserves_inactive_positions_and_final_observations(
    device: torch.device, compiled: bool, record_trace: bool
) -> None:
    values = torch.tensor([4.0, 0.0, 1.0, 3.0], device=device)
    enabled = torch.tensor([False, True, True, True], device=device)

    def solve(values: Tensor):
        return _count_down(values, enabled, 5, record_trace)

    run = torch.compile(solve, fullgraph=True) if compiled else solve
    state, trace = run(values)
    torch.testing.assert_close(state.value, torch.tensor([4.0, 0.0, 0.0, 0.0], device=device))
    assert not state.is_active.any()
    if trace is None:
        assert not record_trace
    else:
        expected = torch.tensor(
            [
                [torch.nan] * 5,
                [torch.nan] * 5,
                [0.0, torch.nan, torch.nan, torch.nan, torch.nan],
                [2.0, 1.0, 0.0, torch.nan, torch.nan],
            ],
            device=device,
        )
        torch.testing.assert_close(trace.value, expected, equal_nan=True)
        torch.testing.assert_close(trace.select(0).value, expected[:, 0], equal_nan=True)


@pytest.mark.parametrize("max_iter", [0, -1])
def test_while_requires_positive_iteration_capacity(device: torch.device, max_iter: int) -> None:
    with pytest.raises(ValueError, match="max_iter must be positive"):
        _count_down(torch.ones(2, device=device), torch.ones(2, dtype=torch.bool, device=device), max_iter, False)


@pytest.mark.parametrize("record_trace", [False, True])
def test_non_strict_cap_returns_unconverged_state(device: torch.device, record_trace: bool) -> None:
    state, trace = _count_down(
        torch.tensor([4.0, 1.0], device=device),
        torch.ones(2, dtype=torch.bool, device=device),
        2,
        record_trace,
        strict=False,
    )
    torch.testing.assert_close(state.value, torch.tensor([2.0, 0.0], device=device))
    assert state.is_active.tolist() == [True, False]
    if trace is not None:
        assert trace.value[1, 1].isnan()


@pytest.mark.parametrize("record_trace", [False, True])
def test_strict_iteration_cap_rejects_an_unconverged_state(record_trace: bool) -> None:
    # CUDA assertions are asynchronous and poison the process; verify this contract on CPU.
    with pytest.raises(RuntimeError, match="did not converge"):
        _count_down(torch.tensor([4.0]), torch.tensor([True]), 2, record_trace)


def test_inactive_initial_state_has_only_nan_history(device: torch.device) -> None:
    state, trace = _count_down(torch.ones(2, device=device), torch.zeros(2, dtype=torch.bool, device=device), 3, True)
    torch.testing.assert_close(state.value, torch.ones(2, device=device))
    assert trace.value.isnan().all()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("initial_value", [0.0, 2.0])
def test_scan_skips_inactive_body_and_uses_default_trace(compiled: bool, initial_value: float) -> None:
    # A runtime assertion proves skipping; Python call counts also count tracing.
    def run(value: Tensor):
        initial = _State(value=value, is_active=value > 0)

        def evaluate(current: _State):
            torch_assert_async(current.is_active.any(), "Inactive numerical body executed")
            next_value = current.value - 1
            return _State(value=next_value, is_active=next_value > 0), _Trace(value=next_value)

        return run_solving_trace_scan(
            init_state=initial,
            body_fn=evaluate,
            default_trace=_Trace(value=torch.full_like(value, -7)),
            max_iter=4,
            strict=True,
        )

    # This sentinel makes default reuse observable independently of NaN masking.
    execute = torch.compile(run, fullgraph=True) if compiled else run
    state, trace = execute(torch.tensor([initial_value]))
    torch.testing.assert_close(state.value, torch.zeros(1))
    expected = [1.0, 0.0, -7.0, -7.0] if initial_value else [-7.0] * 4
    torch.testing.assert_close(trace.value, torch.tensor([expected]))


def test_empty_trace_selection_does_not_stop_active_solving(device: torch.device) -> None:
    values = torch.tensor([1.0, 3.0], device=device)

    def evaluate(current: _State):
        next_value = torch.where(current.is_active, current.value - 1, current.value)
        return _State(value=next_value, is_active=next_value > 0), _Trace(value=next_value)

    def run():
        return run_solving_trace_scan(
            init_state=_State(value=values, is_active=values > 0),
            body_fn=evaluate,
            default_trace=_Trace(value=torch.full_like(values, torch.nan)),
            max_iter=5,
            strict=True,
            trace_mask=torch.zeros_like(values, dtype=torch.bool),
        )

    state, trace = torch.compile(run, fullgraph=True)()
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    assert trace.value.isnan().all()


@pytest.mark.parametrize("counts", [[1.0, 2.0], [1.0]])
def test_nested_scan_preserves_position_and_iteration_axes(device: torch.device, counts: list[float]) -> None:
    values = torch.tensor(counts, device=device)
    initial = _State(value=values, is_active=torch.ones_like(values, dtype=torch.bool))

    def evaluate(state: _State):
        _, inner = _count_down(state.value, state.is_active, 3, True)
        next_value = torch.where(state.is_active, state.value - 1, state.value)
        return _State(value=next_value, is_active=state.is_active & (next_value > 0)), _NestedTrace(
            value=next_value, inner=inner
        )

    def run():
        return run_solving_trace_scan(
            init_state=initial,
            body_fn=evaluate,
            default_trace=_NestedTrace(
                value=torch.full_like(values, torch.nan),
                inner=_Trace(value=torch.full((len(counts), 3), torch.nan, device=device)),
            ),
            max_iter=3,
            strict=True,
        )

    state, trace = torch.compile(run, fullgraph=True)()
    assert trace.value.shape == (len(counts), 3)
    assert trace.inner.value.shape == (len(counts), 3, 3)
    assert trace.inner.value[0, :, 1:].isnan().all()
    torch.testing.assert_close(trace.inner.select(0).select(0).value, values - 1)
    assert not state.is_active.any()


def test_scan_graph_carries_only_state_and_skips_inactive_evaluation(device: torch.device) -> None:
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    values = torch.tensor([2.0, 1.0], device=device)

    def run():
        return _count_down(values, torch.ones_like(values, dtype=torch.bool), 4, True)

    torch.compile(run, backend=backend, fullgraph=True)()
    nodes = [
        node
        for module in graphs[0].modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    ]
    scans = [node for node in nodes if node.target is torch.ops.higher_order.scan]
    assert len(scans) == 1
    assert len(scans[0].args[1]) == 2
    assert all(node.target is not torch.ops.higher_order.while_loop for node in nodes)
    assert sum(node.target is torch.ops.higher_order.cond for node in nodes) == 1


def test_scan_cuda_peak_does_not_duplicate_complete_history(device: torch.device) -> None:
    if device.type != "cuda":
        pytest.skip("CUDA allocation statistics require a CUDA device")
    width, length = 1048576, 16
    row = torch.linspace(0, 1, width, device=device)
    initial = _State(value=torch.zeros((), device=device), is_active=torch.ones((), dtype=torch.bool, device=device))
    empty = _Trace(value=torch.full_like(row, torch.nan))

    def evaluate(state):
        return _State(value=state.value + 1, is_active=state.value < length - 1), _Trace(value=row + state.value)

    def run():
        return run_solving_trace_scan(
            init_state=initial, body_fn=evaluate, default_trace=empty, max_iter=length, strict=True
        )

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        warmup = compiled()
        torch.cuda.synchronize(device)
        del warmup
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        _, trace = compiled()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline
    assert peak < 1.5 * width * length * row.element_size(), peak
    torch.testing.assert_close(trace.value[:, -1], row + length - 1)


class _MaskTrace(SolvingTrace):
    value: Tensor
    limited: Tensor
    count: Tensor
    inner: _Trace
    optional: _Trace | None


@pytest.mark.parametrize("compiled", [False, True])
def test_trace_mask_preserves_nested_axes_dtypes_and_existing_nan(device: torch.device, compiled: bool) -> None:
    value = torch.ones((2, 3), device=device)
    value[0, 1] = torch.nan
    trace = _MaskTrace(
        value=value,
        limited=torch.ones(2, dtype=torch.bool, device=device),
        count=torch.ones((2, 3), dtype=torch.int64, device=device),
        inner=_Trace(value=torch.ones((2, 4, 3), device=device)),
        optional=None,
    )
    valid = torch.tensor([True, False], device=device)

    def apply_mask(trace, valid):
        return trace.mask_invalid(valid=valid)

    run = torch.compile(apply_mask, fullgraph=True) if compiled else apply_mask
    result = run(trace, valid)
    torch.testing.assert_close(result.value[0], value[0], equal_nan=True)
    assert result.value[1].isnan().all()
    assert result.inner.value[1].isnan().all()
    torch.testing.assert_close(result.inner.value[0], trace.inner.value[0])
    torch.testing.assert_close(result.limited, valid)
    torch.testing.assert_close(result.count[1], torch.full((3,), -1, dtype=torch.int64, device=device))
    assert result.optional is None
    assert trace.value[1].isfinite().all()
    assert trace.limited.all()


@pytest.mark.parametrize("record_trace", [False, True])
def test_solving_rejects_nonboolean_activity(device: torch.device, record_trace: bool) -> None:
    state = _State(value=torch.ones(2, device=device), is_active=torch.ones(2, device=device))

    def solve() -> None:
        if record_trace:
            run_solving_trace_scan(
                init_state=state,
                body_fn=lambda state: (state, _Trace(value=state.value)),
                default_trace=_Trace(value=torch.full_like(state.value, torch.nan)),
                max_iter=1,
                strict=False,
            )
        else:
            run_solving_loop(init_state=state, body_fn=lambda state: state, max_iter=1, strict=False)

    with pytest.raises(TypeError, match="is_active must be a boolean mask"):
        solve()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.complex64])
def test_trace_mask_rejects_fields_without_the_declared_unused_value(device: torch.device, dtype: torch.dtype) -> None:
    trace = _Trace(value=torch.ones(2, dtype=dtype, device=device))
    with pytest.raises(TypeError, match="signed integer"):
        trace.mask_invalid(torch.ones(2, dtype=torch.bool, device=device))


@pytest.mark.parametrize("compiled", [False, True])
def test_trace_mask_broadcasts_position_axes(device: torch.device, compiled: bool) -> None:
    trace = _Trace(value=torch.arange(24, dtype=torch.float64, device=device).reshape(2, 3, 4))
    valid = torch.tensor([[True, False, True]], device=device)

    def apply_mask(trace: _Trace, valid: Tensor) -> _Trace:
        return trace.mask_invalid(valid)

    run = torch.compile(apply_mask, fullgraph=True) if compiled else apply_mask
    result = run(trace, valid)
    torch.testing.assert_close(result.value[:, 0], trace.value[:, 0])
    torch.testing.assert_close(result.value[:, 2], trace.value[:, 2])
    assert result.value[:, 1].isnan().all()
    assert trace.value.isfinite().all()


@pytest.mark.parametrize("compiled", [False, True])
def test_trace_selection_broadcasts_without_changing_convergence(device: torch.device, compiled: bool) -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]], device=device).expand(2, 3).clone()
    mask = torch.tensor([[True, False, True]], device=device)

    def body_fn(state: _State) -> tuple[_State, _Trace]:
        value = torch.where(state.is_active, state.value - 1, state.value)
        return _State(value=value, is_active=value > 0), _Trace(value=value)

    def solve(values: Tensor) -> tuple[_State, _Trace]:
        return run_solving_trace_scan(
            init_state=_State(value=values, is_active=values > 0),
            body_fn=body_fn,
            default_trace=_Trace(value=torch.full_like(values, torch.nan)),
            max_iter=3,
            strict=True,
            trace_mask=mask,
        )

    run = torch.compile(solve, fullgraph=True) if compiled else solve
    state, trace = run(values)
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    assert not state.any_active
    expected_counts = torch.tensor([[1, 0, 3]], device=device).expand(2, 3)
    torch.testing.assert_close(trace.value.isfinite().sum(-1), expected_counts)
