"""Counted loops and scan variants preserve state, iteration order, and outputs."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.loop import (
    run_scan,
    run_scan_without_carry,
    run_scan_without_inputs,
    run_scan_without_output,
    run_while_loop_with_counter,
)
from neurox.common.pytree_dataclass_mixin import PyTreeDataClassMixin
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin


class _State(TensorDataClassMixin, PyTreeDataClassMixin):
    value: Tensor


def _count_down(value: Tensor, max_iter: int) -> _State:
    def cond_fn(step: Tensor, state: _State) -> Tensor:
        return (step < max_iter) & (state.value > 0).any()

    def body_fn(_step: Tensor, state: _State) -> _State:
        return _State(value=(state.value - 1).clamp_min(0))

    return run_while_loop_with_counter(
        init_state=_State(value=value),
        cond_fn=cond_fn,
        body_fn=body_fn,
        device=value.device,
    )


@pytest.mark.parametrize("max_iter", [0, 2, 5])
def test_caller_condition_bounds_updates(device: torch.device, max_iter: int) -> None:
    value = torch.tensor([0.0, 1.0, 3.0], device=device, dtype=torch.float64)
    original = value.clone()

    state = _count_down(value, max_iter)

    torch.testing.assert_close(state.value, (original - max_iter).clamp_min(0))
    torch.testing.assert_close(value, original)


def test_body_receives_zero_based_tensor_step(device: torch.device) -> None:
    value = torch.zeros((), device=device, dtype=torch.float64)

    def cond_fn(step: Tensor, _state: _State) -> Tensor:
        return step < 4

    def body_fn(step: Tensor, state: _State) -> _State:
        return _State(value=state.value + step.to(state.value.dtype))

    state = run_while_loop_with_counter(
        init_state=_State(value=value),
        cond_fn=cond_fn,
        body_fn=body_fn,
        device=device,
    )

    torch.testing.assert_close(state.value, torch.tensor(6.0, device=device, dtype=value.dtype))


def test_integer_state_and_boolean_condition() -> None:
    def cond_fn(_step: Tensor, state: int) -> bool:
        return state < 3

    def body_fn(_step: Tensor, state: int) -> int:
        return state + 1

    state = run_while_loop_with_counter(
        init_state=0,
        cond_fn=cond_fn,
        body_fn=body_fn,
        device=torch.device("cpu"),
    )
    assert state == 3


@pytest.mark.parametrize("max_iter", [0, 2, 5])
def test_fullgraph_preserves_structured_state(device: torch.device, max_iter: int) -> None:
    value = torch.tensor([0.0, 1.0, 3.0], device=device)

    def solve(value: Tensor) -> Tensor:
        return _count_down(value, max_iter).value

    expected = solve(value)
    with torch._dynamo.config.patch(disable=False):
        actual = torch.compile(solve, backend="inductor", fullgraph=True)(value)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_with_state_and_inputs(device: torch.device, dim: int, reverse: bool) -> None:
    values = torch.arange(8, device=device, dtype=torch.float64).reshape(4, 2).movedim(0, dim)

    def body_fn(state: _State, item: _State) -> tuple[_State, _State]:
        updated = state.value * 0.5 + item.value
        return _State(value=updated), _State(value=updated.square())

    def run(values: Tensor) -> tuple[_State, _State]:
        return run_scan(
            init_state=_State(value=values.new_zeros(2)),
            xs=_State(value=values),
            body_fn=body_fn,
            output_template=_State(value=values.new_empty(0)),
            dim=dim,
            reverse=reverse,
        )

    expected_state = values.new_zeros(2)
    history = []
    for index in reversed(range(4)) if reverse else range(4):
        expected_state = expected_state * 0.5 + values.select(dim, index)
        history.append(expected_state.square())
    expected_output = torch.stack(list(reversed(history)) if reverse else history, dim=dim)

    for state, output in (run(values), torch.compile(run, fullgraph=True)(values)):
        torch.testing.assert_close(state.value, expected_state)
        torch.testing.assert_close(output.value, expected_output)


@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_without_output_returns_only_final_state(
    device: torch.device, length: int, dim: int, reverse: bool
) -> None:
    values = torch.arange(length * 2, device=device, dtype=torch.float64).reshape(length, 2).movedim(0, dim)
    original = values.clone()
    initial = values.new_ones(2)

    def body_fn(state: _State, item: _State) -> _State:
        return _State(value=state.value * 0.5 + item.value)

    def run(initial: Tensor, values: Tensor) -> _State:
        return run_scan_without_output(
            init_state=_State(value=initial),
            xs=_State(value=values),
            body_fn=body_fn,
            device=device,
            dim=dim,
            reverse=reverse,
        )

    expected = initial
    for index in reversed(range(length)) if reverse else range(length):
        expected = expected * 0.5 + values.select(dim, index)

    for state in (run(initial, values), torch.compile(run, fullgraph=True)(initial, values)):
        assert type(state) is _State
        torch.testing.assert_close(state.value, expected)
    torch.testing.assert_close(values, original)
    torch.testing.assert_close(initial, values.new_ones(2))


@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_without_inputs_updates_state_exactly_length_times(
    device: torch.device, length: int, reverse: bool
) -> None:
    initial = torch.tensor([1.0, 3.0], device=device, dtype=torch.float64)

    def body_fn(state: _State) -> tuple[_State, _State]:
        updated = state.value * 2
        return _State(value=updated), _State(value=updated.square())

    def run(initial: Tensor) -> tuple[_State, _State]:
        return run_scan_without_inputs(
            init_state=_State(value=initial),
            length=length,
            body_fn=body_fn,
            output_template=_State(value=initial.new_empty(0)),
            device=device,
            reverse=reverse,
        )

    expected_state = initial * 2**length
    expected_output = torch.stack([(initial * 2**step).square() for step in range(1, length + 1)])
    if reverse:
        expected_output = expected_output.flip(0)

    for state, output in (run(initial), torch.compile(run, fullgraph=True)(initial)):
        torch.testing.assert_close(state.value, expected_state)
        torch.testing.assert_close(output.value, expected_output)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_without_carry_returns_only_collected_outputs(device: torch.device, dim: int, reverse: bool) -> None:
    values = torch.arange(8, device=device, dtype=torch.float64).reshape(4, 2).movedim(0, dim)

    def body_fn(item: _State) -> _State:
        return _State(value=item.value.square() + 1)

    def run(values: Tensor) -> _State:
        return run_scan_without_carry(
            xs=_State(value=values),
            body_fn=body_fn,
            output_template=_State(value=values.new_empty(0)),
            device=device,
            dim=dim,
            reverse=reverse,
        )

    for output in (run(values), torch.compile(run, fullgraph=True)(values)):
        assert type(output) is _State
        torch.testing.assert_close(output.value, values.square() + 1)
