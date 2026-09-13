"""Shared numerical steps, convergence masks, and scan history storage."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch
from torch import Tensor
from torch.utils import _pytree as pytree

from neurox.common.torch_compat import torch_assert_async
from neurox.execution.solving import SolvingState, SolvingTrace, run_solving_loop, run_solving_trace_scan


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


@pytest.mark.parametrize("record_trace", [False, True])
def test_shared_step_preserves_inactive_positions_and_final_observations(
    device: torch.device, record_trace: bool
) -> None:
    values = torch.tensor([4.0, 0.0, 1.0, 3.0], device=device)
    enabled = torch.tensor([False, True, True, True], device=device)

    def solve(values: Tensor):
        return _count_down(values, enabled, 5, record_trace)

    state, trace = solve(values)
    torch.testing.assert_close(state.value, torch.tensor([4.0, 0.0, 0.0, 0.0], device=device))
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


@pytest.mark.parametrize("active", [False, True])
def test_scan_skips_inactive_body_and_uses_default_trace(device: torch.device, active: bool) -> None:
    # A runtime assertion proves skipping; Python call counts also count tracing.
    def run(value: Tensor):
        initial = _State(value=value, is_active=value > 0)

        def evaluate(current: _State):
            torch_assert_async(current.is_active.any(), "Inactive numerical body executed")
            next_value = torch.where(current.is_active, current.value - 1, current.value)
            return _State(value=next_value, is_active=next_value > 0), _Trace(value=next_value)

        return run_solving_trace_scan(
            init_state=initial,
            body_fn=evaluate,
            default_trace=_Trace(value=torch.full_like(value, -7)),
            max_iter=4,
            strict=True,
        )

    # This sentinel makes default reuse observable independently of NaN masking.
    values = torch.tensor([2.0, 1.0] if active else [0.0, 0.0], device=device)
    state, trace = run(values)
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    expected = [[1.0, 0.0, -7.0, -7.0], [0.0, torch.nan, -7.0, -7.0]] if active else [[-7.0] * 4] * 2
    torch.testing.assert_close(trace.value, torch.tensor(expected, device=device), equal_nan=True)


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

    state, trace = run()
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

    _state, trace = run()
    assert trace.value.shape == (len(counts), 3)
    assert trace.inner.value.shape == (len(counts), 3, 3)
    assert trace.inner.value[0, :, 1:].isnan().all()
    torch.testing.assert_close(trace.inner.select(0).select(0).value, values - 1)


class _MaskTrace(SolvingTrace):
    value: Tensor
    limited: Tensor
    count: Tensor
    inner: _Trace
    optional: _Trace | None


def test_trace_mask_preserves_nested_axes_dtypes_and_existing_nan(device: torch.device) -> None:
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

    result = apply_mask(trace, valid)
    torch.testing.assert_close(result.value[0], value[0], equal_nan=True)
    assert result.value[1].isnan().all()
    assert result.inner.value[1].isnan().all()
    torch.testing.assert_close(result.inner.value[0], trace.inner.value[0])
    torch.testing.assert_close(result.limited, valid)
    torch.testing.assert_close(result.count[1], torch.full((3,), -1, dtype=torch.int64, device=device))
    assert result.optional is None
    assert trace.value[1].isfinite().all()
    assert trace.limited.all()


def test_state_reconstruction_validates_activity(device: torch.device) -> None:
    state = _State(value=torch.ones(2, device=device), is_active=torch.ones(2, dtype=torch.bool, device=device))
    invalid = torch.ones(2, device=device)
    with pytest.raises(TypeError, match="boolean mask"):
        replace(state, is_active=invalid)
    leaves, spec = pytree.tree_flatten(state)
    restored = pytree.tree_unflatten(leaves, spec)
    torch.testing.assert_close(restored.is_active, state.is_active)
    with pytest.raises(TypeError, match="boolean mask"):
        pytree.tree_unflatten([leaf.float() if leaf.dtype == torch.bool else leaf for leaf in leaves], spec)


def test_trace_reconstruction_validates_replacement_tensors(device: torch.device) -> None:
    trace = _Trace(value=torch.ones(2, device=device))
    invalid = torch.ones(2, dtype=torch.uint8, device=device)
    with pytest.raises(TypeError, match="signed integer"):
        replace(trace, value=invalid)
    leaves, spec = pytree.tree_flatten(trace)
    torch.testing.assert_close(pytree.tree_unflatten(leaves, spec).value, trace.value)
    with pytest.raises(TypeError, match="signed integer"):
        pytree.tree_unflatten([invalid], spec)


def test_trace_construction_validates_nested_dataclass_fields(device: torch.device) -> None:
    @dataclass
    class _Observation:
        value: Tensor

    class _ContainerTrace(SolvingTrace):
        observation: _Observation

    observation = _Observation(value=torch.ones(2, dtype=torch.uint8, device=device))
    with pytest.raises(TypeError, match="signed integer"):
        _ContainerTrace(observation=observation)


def test_trace_mask_broadcasts_position_axes(device: torch.device) -> None:
    trace = _Trace(value=torch.arange(24, dtype=torch.float64, device=device).reshape(2, 3, 4))
    valid = torch.tensor([[True, False, True]], device=device)

    def apply_mask(trace: _Trace, valid: Tensor) -> _Trace:
        return trace.mask_invalid(valid)

    result = apply_mask(trace, valid)
    torch.testing.assert_close(result.value[:, 0], trace.value[:, 0])
    torch.testing.assert_close(result.value[:, 2], trace.value[:, 2])
    assert result.value[:, 1].isnan().all()
    assert trace.value.isfinite().all()


def test_trace_selection_broadcasts_without_changing_convergence(device: torch.device) -> None:
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

    state, trace = solve(values)
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    expected_counts = torch.tensor([[1, 0, 3]], device=device).expand(2, 3)
    torch.testing.assert_close(trace.value.isfinite().sum(-1), expected_counts)
