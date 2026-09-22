"""Shared numerical steps, convergence masks, and scan history storage."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

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

    state, trace = _count_down(values, enabled, 5, record_trace)
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


@pytest.mark.parametrize("active", [False, True])
def test_scan_skips_inactive_body_and_uses_default_trace(device: torch.device, active: bool) -> None:
    # A runtime assertion proves skipping; Python call counts also count tracing.
    def evaluate(current: _State):
        torch_assert_async(current.is_active.any(), "Inactive numerical body executed")
        next_value = torch.where(current.is_active, current.value - 1, current.value)
        return _State(value=next_value, is_active=next_value > 0), _Trace(value=next_value)

    # This sentinel makes default reuse observable independently of NaN masking.
    values = torch.tensor([2.0, 1.0] if active else [0.0, 0.0], device=device)
    state, trace = run_solving_trace_scan(
        init_state=_State(value=values, is_active=values > 0),
        body_fn=evaluate,
        default_trace=_Trace(value=torch.full_like(values, -7)),
        max_iter=4,
        strict=True,
    )
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    expected = [[1.0, 0.0, -7.0, -7.0], [0.0, torch.nan, -7.0, -7.0]] if active else [[-7.0] * 4] * 2
    torch.testing.assert_close(trace.value, torch.tensor(expected, device=device), equal_nan=True)


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

    _state, trace = run_solving_trace_scan(
        init_state=initial,
        body_fn=evaluate,
        default_trace=_NestedTrace(
            value=torch.full_like(values, torch.nan),
            inner=_Trace(value=torch.full((len(counts), 3), torch.nan, device=device)),
        ),
        max_iter=3,
        strict=True,
    )
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
    value = torch.ones((2, 3, 4), device=device)
    value[0, 0, 1] = torch.nan
    trace = _MaskTrace(
        value=value,
        limited=torch.ones((2, 3), dtype=torch.bool, device=device),
        count=torch.ones((2, 3, 4), dtype=torch.int64, device=device),
        inner=_Trace(value=torch.ones((2, 3, 5, 4), device=device)),
        optional=None,
    )
    valid = torch.tensor([[True, False, True]], device=device)
    result = trace.mask_invalid(valid)
    torch.testing.assert_close(result.value[:, [0, 2]], value[:, [0, 2]], equal_nan=True)
    assert result.value[:, 1].isnan().all()
    assert result.inner.value[:, 1].isnan().all()
    torch.testing.assert_close(result.inner.value[:, [0, 2]], trace.inner.value[:, [0, 2]])
    torch.testing.assert_close(result.limited, valid.expand(2, 3))
    torch.testing.assert_close(result.count[:, 1], torch.full((2, 4), -1, dtype=torch.int64, device=device))
    assert result.optional is None
    assert trace.value[:, 1].isfinite().all()
    assert trace.limited.all()


@pytest.mark.parametrize(
    ("selection", "expected_counts"),
    [([False, False, False], [0, 0, 0]), ([True, False, True], [1, 0, 3])],
    ids=["empty", "partial"],
)
def test_trace_selection_broadcasts_without_changing_convergence(
    device: torch.device, selection: list[bool], expected_counts: list[int]
) -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]], device=device).expand(2, 3).clone()
    mask = torch.tensor([selection], device=device)

    def body_fn(state: _State) -> tuple[_State, _Trace]:
        value = torch.where(state.is_active, state.value - 1, state.value)
        return _State(value=value, is_active=value > 0), _Trace(value=value)

    state, trace = run_solving_trace_scan(
        init_state=_State(value=values, is_active=values > 0),
        body_fn=body_fn,
        default_trace=_Trace(value=torch.full_like(values, torch.nan)),
        max_iter=5,
        strict=True,
        trace_mask=mask,
    )
    torch.testing.assert_close(state.value, torch.zeros_like(values))
    torch.testing.assert_close(
        trace.value.isfinite().sum(-1), torch.tensor([expected_counts], device=device).expand(2, 3)
    )
    assert trace.value[:, ~mask[0]].isnan().all()
