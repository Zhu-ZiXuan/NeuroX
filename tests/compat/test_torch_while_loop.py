"""Compatibility contract tests for the public `torch.while_loop` API."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

_MAX_STEPS = 6
_STOP_ABS = 0.0625

_Trace = tuple[Tensor, Tensor]
_UntracedResult = tuple[Tensor, Tensor, Tensor]
_TracedResult = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def _transition(values: Tensor, active: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    candidate = values * 0.5
    next_values = torch.where(active, candidate, values)
    next_active = active & (candidate.abs() > _STOP_ABS)
    return next_values, next_active, candidate.abs().amax()


def _loop_inputs(values: Tensor, max_steps: int) -> tuple[Tensor, Tensor, Tensor]:
    step = torch.zeros((), dtype=torch.int64, device=values.device)
    limit = torch.full((), max_steps, dtype=torch.int64, device=values.device)
    active = values.abs() > _STOP_ABS
    return step, limit, active


def _solve_without_trace(values: Tensor, max_steps: int) -> _UntracedResult:
    step, limit, active = _loop_inputs(values, max_steps)

    def cond_fn(step: Tensor, state: Tensor, active: Tensor) -> Tensor:
        del state
        return (step < limit) & torch.any(active)

    def body_fn(step: Tensor, state: Tensor, active: Tensor) -> _UntracedResult:
        next_values, next_active, _ = _transition(state, active)
        return step + 1, next_values, next_active

    return torch.while_loop(cond_fn, body_fn, (step, values, active))


def _empty_trace(values: Tensor, max_steps: int) -> _Trace:
    residual = torch.full(
        (max_steps,),
        torch.nan,
        dtype=values.dtype,
        device=values.device,
    )
    active_count = torch.zeros(max_steps, dtype=torch.int64, device=values.device)
    return residual, active_count


def _solve_with_trace(values: Tensor, max_steps: int) -> _TracedResult:
    step, limit, active = _loop_inputs(values, max_steps)
    residual_trace, active_count_trace = _empty_trace(values, max_steps)

    def cond_fn(
        step: Tensor,
        state: Tensor,
        active: Tensor,
        residual_trace: Tensor,
        active_count_trace: Tensor,
    ) -> Tensor:
        del state, residual_trace, active_count_trace
        return (step < limit) & torch.any(active)

    def body_fn(
        step: Tensor,
        state: Tensor,
        active: Tensor,
        residual_trace: Tensor,
        active_count_trace: Tensor,
    ) -> _TracedResult:
        next_values, next_active, residual = _transition(state, active)
        index = step.unsqueeze(0)
        next_residual_trace = residual_trace.scatter(0, index, residual.unsqueeze(0))
        next_active_count_trace = active_count_trace.scatter(
            0,
            index,
            next_active.sum(dtype=torch.int64).unsqueeze(0),
        )
        return (
            step + 1,
            next_values,
            next_active,
            next_residual_trace,
            next_active_count_trace,
        )

    return torch.while_loop(
        cond_fn,
        body_fn,
        (step, values, active, residual_trace, active_count_trace),
    )


def _adaptive_solve(
    values: Tensor,
    max_steps: int,
    record_trace: bool,
) -> _UntracedResult | _TracedResult:
    # The Python boolean deliberately specializes observation work at capture time.
    if record_trace:
        return _solve_with_trace(values, max_steps)
    return _solve_without_trace(values, max_steps)


def _numerical_result(result: _UntracedResult | _TracedResult) -> _UntracedResult:
    return result[0], result[1], result[2]


def _assert_numerical_equal(
    actual: _UntracedResult | _TracedResult,
    expected: _UntracedResult | _TracedResult,
) -> None:
    actual_step, actual_values, actual_active = _numerical_result(actual)
    expected_step, expected_values, expected_active = _numerical_result(expected)
    assert torch.equal(actual_step, expected_step)
    torch.testing.assert_close(actual_values, expected_values)
    assert torch.equal(actual_active, expected_active)


def test_device_local_counter_and_tensor_exit(device: torch.device) -> None:
    values = torch.tensor([1.0, 0.125], device=device)

    step, solved, active = _solve_without_trace(values, _MAX_STEPS)

    assert step.ndim == 0
    assert step.dtype == torch.int64
    assert step.device == values.device
    assert int(step) == 4
    torch.testing.assert_close(solved, torch.full_like(values, _STOP_ABS))
    assert not torch.any(active)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_carried_tensor_metadata_stays_stable(device: torch.device, dtype: torch.dtype) -> None:
    values = torch.ones((2, 3), dtype=dtype, device=device)

    _, solved, active = _solve_without_trace(values, 1)

    assert solved.shape == values.shape
    assert solved.dtype == values.dtype
    assert solved.device == values.device
    assert solved.stride() == values.stride()
    assert active.shape == values.shape
    assert active.dtype == torch.bool
    assert active.device == values.device
    assert active.stride() == values.stride()


@pytest.mark.parametrize("record_trace", [False, True], ids=["trace_off", "trace_on"])
def test_direct_and_caller_compiled_use_the_same_function(
    device: torch.device,
    record_trace: bool,
) -> None:
    values = torch.tensor([1.0, 0.125], device=device)

    direct = _adaptive_solve(values, _MAX_STEPS, record_trace)
    compiled = torch.compile(_adaptive_solve, backend="eager", fullgraph=True)
    captured = compiled(values, _MAX_STEPS, record_trace)

    _assert_numerical_equal(captured, direct)
    assert len(captured) == (5 if record_trace else 3)


def test_trace_flag_changes_only_observation_work(device: torch.device) -> None:
    values = torch.tensor([1.0, 0.125], device=device)

    untraced = _adaptive_solve(values, _MAX_STEPS, False)
    traced = _adaptive_solve(values, _MAX_STEPS, True)

    _assert_numerical_equal(traced, untraced)
    assert len(untraced) == 3
    assert len(traced) == 5
    residual_trace = traced[3]
    active_count_trace = traced[4]
    expected_residual = torch.tensor(
        [0.5, 0.25, 0.125, _STOP_ABS],
        dtype=values.dtype,
        device=device,
    )
    torch.testing.assert_close(residual_trace[:4], expected_residual)
    assert torch.isnan(residual_trace[4:]).all()
    assert torch.equal(
        active_count_trace,
        torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.int64, device=device),
    )


def test_zero_step_and_limit_termination(device: torch.device) -> None:
    converged = torch.tensor([_STOP_ABS], device=device)
    unconverged = torch.tensor([1.0], device=device)

    zero_result = _solve_with_trace(converged, 0)
    cap_result = _solve_without_trace(unconverged, 2)

    zero_step, zero_values, zero_active, zero_residual_trace, zero_active_count_trace = zero_result
    cap_step, cap_values, cap_active = cap_result
    assert int(zero_step) == 0
    assert torch.equal(zero_values, converged)
    assert not torch.any(zero_active)
    assert zero_residual_trace.numel() == 0
    assert zero_active_count_trace.numel() == 0
    assert int(cap_step) == 2
    torch.testing.assert_close(cap_values, torch.tensor([0.25], device=device))
    assert torch.all(cap_active)


def _nested_solve(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    outer_step = torch.zeros((), dtype=torch.int64, device=values.device)
    outer_limit = torch.full((), 3, dtype=torch.int64, device=values.device)
    inner_limit = torch.full((), 2, dtype=torch.int64, device=values.device)
    total_inner = torch.zeros((), dtype=torch.int64, device=values.device)

    def outer_cond(outer_step: Tensor, state: Tensor, total_inner: Tensor) -> Tensor:
        del state, total_inner
        return outer_step < outer_limit

    def outer_body(
        outer_step: Tensor,
        state: Tensor,
        total_inner: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        inner_step = torch.zeros((), dtype=torch.int64, device=state.device)

        def inner_cond(inner_step: Tensor, state: Tensor, count: Tensor) -> Tensor:
            del state, count
            return inner_step < inner_limit

        def inner_body(
            inner_step: Tensor,
            state: Tensor,
            count: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            return inner_step + 1, state * 0.5, count + 1

        _, inner_values, next_total_inner = torch.while_loop(
            inner_cond,
            inner_body,
            (inner_step, state, total_inner),
        )
        return outer_step + 1, inner_values, next_total_inner

    return torch.while_loop(
        outer_cond,
        outer_body,
        (outer_step, values, total_inner),
    )


def test_nested_while_loop(device: torch.device) -> None:
    values = torch.ones(4, device=device)

    step, solved, total_inner = _nested_solve(values)

    assert int(step) == 3
    assert int(total_inner) == 6
    torch.testing.assert_close(solved, torch.full_like(values, 1.0 / 64.0))


class _BufferedDecay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("decay", torch.tensor(0.5))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        step = torch.zeros((), dtype=torch.int64, device=values.device)
        limit = torch.full((), 3, dtype=torch.int64, device=values.device)

        def cond_fn(step: Tensor, state: Tensor) -> Tensor:
            del state
            return step < limit

        def body_fn(step: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
            return step + 1, state * self.decay

        return torch.while_loop(cond_fn, body_fn, (step, values))


def test_module_buffer_is_lifted_into_fullgraph(device: torch.device) -> None:
    module = _BufferedDecay().to(device)
    values = torch.ones(4, device=device)

    compiled = torch.compile(module, backend="eager", fullgraph=True)
    step, solved = compiled(values)

    assert int(step) == 3
    torch.testing.assert_close(solved, torch.full_like(values, 0.125))


def test_fullgraph_capture_has_one_outer_graph(device: torch.device) -> None:
    captured_graphs: list[torch.fx.GraphModule] = []

    def backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[object],
    ) -> Callable[..., object]:
        captured_graphs.append(graph_module)
        return graph_module.forward

    values = torch.tensor([1.0, 0.125], device=device)

    compiled = torch.compile(
        _solve_without_trace,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    result = compiled(values, _MAX_STEPS)

    assert int(result[0]) == 4
    assert len(captured_graphs) == 1
    assert any(
        "while_loop" in str(node.target) for node in captured_graphs[0].graph.nodes if node.op == "call_function"
    )


@pytest.mark.parametrize("record_trace", [False, True], ids=["trace_off", "trace_on"])
def test_inductor_executes_fullgraph_on_cpu(record_trace: bool) -> None:
    values = torch.tensor([1.0, 0.125])

    expected = _adaptive_solve(values, _MAX_STEPS, record_trace)
    compiled = torch.compile(
        _adaptive_solve,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )
    actual = compiled(values, _MAX_STEPS, record_trace)

    _assert_numerical_equal(actual, expected)
    assert len(actual) == (5 if record_trace else 3)
