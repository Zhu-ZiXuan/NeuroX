"""PyTorch boundary adapters preserve callable types and structured loop states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin
from neurox.common.torch_compat import (
    torch_cond,
    torch_map,
    torch_scan,
    torch_while_loop,
)


class _Iteration(NamedTuple):
    step: Tensor
    active: Tensor


class _Values(NamedTuple):
    current: Tensor
    total: Tensor


class _State(NamedTuple):
    iteration: _Iteration
    values: _Values


@dataclass(frozen=True)
class _PlainDataClassState:
    step: int
    values: Tensor


torch.export.register_dataclass(_PlainDataClassState)


class _DataClassState(TensorDataClassMixin, PyTreeDataClassMixin):
    step: Tensor
    active: Tensor
    values: Tensor


@dataclass(frozen=True)
class _ScanCarry:
    total: Tensor
    steps: Tensor
    optional: Tensor | None = None


torch.export.register_dataclass(_ScanCarry)


class _ScanInput(TensorDataClassMixin, PyTreeDataClassMixin):
    values: Tensor
    scales: Tensor


class _ScanOutput(TensorDataClassMixin, PyTreeDataClassMixin):
    values: _Values
    steps: Tensor


class _MapOutput(TensorDataClassMixin, PyTreeDataClassMixin):
    values: _ScanCarry


@pytest.mark.parametrize("record_optional", [False, True])
def test_map_preserves_structured_inputs_outputs_and_captured_constants(
    device: torch.device, record_optional: bool
) -> None:
    def evaluate(inputs: _ScanInput, constants: dict[str, Tensor], gain: float) -> _MapOutput:
        return _MapOutput(
            values=_ScanCarry(
                total=inputs.values * inputs.scales * gain + constants["bias"],
                steps=inputs.values.sum(),
                optional=inputs.values.square() if record_optional else None,
            ),
        )

    def run(inputs: _ScanInput, constants: dict[str, Tensor]) -> _MapOutput:
        def body(inputs: _ScanInput) -> _MapOutput:
            return evaluate(inputs, constants, 2.5)

        template = _MapOutput(
            values=_ScanCarry(
                total=inputs.values,
                steps=inputs.scales,
                optional=inputs.values if record_optional else None,
            ),
        )
        return torch_map(body, inputs, output_template=template)

    inputs = _ScanInput(
        values=torch.arange(12.0, device=device).reshape(4, 3),
        scales=torch.linspace(1.0, 2.0, 4, device=device).unsqueeze(-1),
    )
    constants = {"bias": torch.tensor([0.5, 1.0, 1.5], device=device)}
    actual = run(inputs, constants)
    expected = [
        evaluate(_ScanInput(values=inputs.values[i], scales=inputs.scales[i]), constants, 2.5) for i in range(4)
    ]
    assert isinstance(actual, _MapOutput)
    assert isinstance(actual.values, _ScanCarry)
    torch.testing.assert_close(actual.values.total, torch.stack([item.values.total for item in expected]))
    torch.testing.assert_close(actual.values.steps, torch.stack([item.values.steps for item in expected]))
    if record_optional:
        torch.testing.assert_close(actual.values.optional, inputs.values.square())
    else:
        assert actual.values.optional is None


def test_cond_preserves_registered_optional_output_and_runtime_selection(device: torch.device) -> None:
    def run(pred: Tensor, state: _ScanCarry) -> _ScanCarry:
        def true_fn(operand: _ScanCarry) -> _ScanCarry:
            return _ScanCarry(total=operand.total + 1, steps=operand.steps + 1)

        def false_fn(operand: _ScanCarry) -> _ScanCarry:
            return _ScanCarry(total=operand.total - 1, steps=operand.steps + 2)

        return torch_cond(pred, true_fn, false_fn, state, output_template=state)

    initial = _ScanCarry(total=torch.ones(3, device=device), steps=torch.zeros((), device=device))
    for selected in (True, False):
        result = run(torch.tensor(selected, device=device), initial)
        torch.testing.assert_close(result.total, initial.total + (1 if selected else -1))
        torch.testing.assert_close(result.steps, initial.steps + (1 if selected else 2))
        assert result.optional is None


def _initial_state(values: Tensor) -> _State:
    return _State(
        iteration=_Iteration(
            step=values.new_zeros((), dtype=torch.int64),
            active=values > 0.125,
        ),
        values=_Values(current=values, total=torch.zeros_like(values)),
    )


def _solve(values: Tensor) -> _State:
    state = _initial_state(values)
    limit = state.iteration.step.new_tensor(8)

    def cond_fn(current: _State) -> Tensor:
        return (current.iteration.step < limit) & torch.any(current.iteration.active)

    def body_fn(current: _State) -> _State:
        candidate = current.values.current * 0.5
        next_values = torch.where(current.iteration.active, candidate, current.values.current)
        return _State(
            iteration=_Iteration(
                step=current.iteration.step + 1,
                active=current.iteration.active & (candidate > 0.125),
            ),
            values=_Values(
                current=next_values,
                total=current.values.total + next_values,
            ),
        )

    return torch_while_loop(cond_fn, body_fn, state)


def _solve_dataclass(values: Tensor) -> _DataClassState:
    state = _DataClassState(
        step=values.new_zeros((), dtype=torch.int64),
        active=values > 0.125,
        values=values,
    )
    limit = state.step.new_tensor(8)

    def cond_fn(current: _DataClassState) -> Tensor:
        return (current.step < limit) & torch.any(current.active)

    def body_fn(current: _DataClassState) -> _DataClassState:
        candidate = current.values * 0.5
        return _DataClassState(
            step=current.step + 1,
            active=current.active & (candidate > 0.125),
            values=torch.where(current.active, candidate, current.values),
        )

    return torch_while_loop(cond_fn, body_fn, state)


def test_nested_namedtuple_state_is_reconstructed() -> None:
    result = _solve(torch.tensor([1.0, 0.125]))

    assert isinstance(result, _State)
    assert isinstance(result.iteration, _Iteration)
    assert isinstance(result.values, _Values)
    assert int(result.iteration.step) == 3
    assert not torch.any(result.iteration.active)
    torch.testing.assert_close(result.values.current, torch.tensor([0.125, 0.125]))
    torch.testing.assert_close(result.values.total, torch.tensor([0.875, 0.375]))


def test_zero_iteration_preserves_state_structure_and_values() -> None:
    initial = _initial_state(torch.tensor([0.125, 0.0]))

    result = torch_while_loop(
        lambda current: torch.any(current.iteration.active),
        lambda current: _initial_state(current.values.current + 1),
        initial,
    )

    assert isinstance(result, _State)
    assert isinstance(result.iteration, _Iteration)
    assert isinstance(result.values, _Values)
    assert torch.equal(result.iteration.step, initial.iteration.step)
    assert torch.equal(result.iteration.active, initial.iteration.active)
    torch.testing.assert_close(result.values.current, initial.values.current)
    torch.testing.assert_close(result.values.total, initial.values.total)


def test_registered_tensor_dataclass_state_is_reconstructed() -> None:
    values = torch.tensor([1.0, 0.125])
    result = _solve_dataclass(values)
    assert type(result) is _DataClassState
    assert int(result.step) == 3
    assert not torch.any(result.active)
    torch.testing.assert_close(result.values, torch.tensor([0.125, 0.125]))


def test_plain_dataclass_state_needs_no_project_base() -> None:
    initial = _PlainDataClassState(step=0, values=torch.ones(2))
    result = torch_while_loop(
        lambda current: current.step < 2,
        lambda current: _PlainDataClassState(step=current.step + 1, values=current.values + 1),
        initial,
    )

    assert type(result) is _PlainDataClassState
    assert result.step == 2
    torch.testing.assert_close(result.values, torch.full((2,), 3.0))


def test_empty_state_preserves_its_structure() -> None:
    result = torch_while_loop(lambda _current: False, lambda current: current, ())

    assert result == ()


def _scan_dataclass(values: Tensor, reverse: bool = False) -> tuple[_ScanCarry, _ScanOutput]:
    def combine_fn(carry: _ScanCarry, item: _ScanInput) -> tuple[_ScanCarry, _ScanOutput]:
        total = carry.total * item.scales + item.values
        steps = carry.steps + 1
        return _ScanCarry(total, steps), _ScanOutput(
            values=_Values(current=item.values.clone(), total=total.clone()), steps=steps.clone()
        )

    return torch_scan(
        combine_fn,
        _ScanCarry(torch.zeros_like(values[0]), values.new_zeros((), dtype=torch.int64)),
        _ScanInput(values=values, scales=values.new_full((values.shape[0],), 0.5)),
        output_template=_ScanOutput(
            values=_Values(values.new_empty(()), values.new_empty(())), steps=values.new_empty(())
        ),
        reverse=reverse,
    )


@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_uses_leading_axis_for_different_rank_leaves(length: int, reverse: bool) -> None:
    values = torch.arange(6 * length, dtype=torch.float32).reshape(length, 2, 3)
    carry = torch.zeros_like(values[0])
    history = []
    indices = range(length)
    for index in reversed(indices) if reverse else indices:
        carry = carry * 0.5 + values[index]
        history.append(carry)
    expected = torch.stack(list(reversed(history)) if reverse else history, dim=0)
    expected_steps = torch.arange(1, length + 1)
    if reverse:
        expected_steps = expected_steps.flip([0])

    final_carry, output = _scan_dataclass(values, reverse)
    assert type(final_carry) is _ScanCarry
    assert final_carry.optional is None
    assert int(final_carry.steps) == length
    assert type(output) is _ScanOutput
    assert type(output.values) is _Values
    torch.testing.assert_close(final_carry.total, carry)
    torch.testing.assert_close(output.values.current, values)
    torch.testing.assert_close(output.values.total, expected)
    torch.testing.assert_close(output.steps, expected_steps)


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_dimension_and_output_layout(dim: int, reverse: bool) -> None:
    def combine_fn(carry: Tensor, item: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        updated = carry * 0.5 + item
        return updated, {"values": updated.clone(), "total": updated.sum()}

    def run(values: Tensor):
        return torch_scan(
            combine_fn,
            values.new_zeros((2, 3)),
            values,
            dim=dim,
            reverse=reverse,
            output_template={"values": values.new_empty(()), "total": values.new_empty(())},
        )

    values = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).movedim(0, dim)
    scan_dim = dim if dim >= 0 else dim + values.ndim
    scan_values = values.movedim(scan_dim, 0)
    carry = values.new_zeros((2, 3))
    output_values = []
    output_totals = []
    indices = range(scan_values.shape[0])
    for index in reversed(indices) if reverse else indices:
        carry, output = combine_fn(carry, scan_values[index])
        output_values.append(output["values"])
        output_totals.append(output["total"])
    if reverse:
        output_values.reverse()
        output_totals.reverse()
    expected_values = torch.stack(output_values)
    if scan_dim < expected_values.ndim:
        expected_values = expected_values.movedim(0, scan_dim)
    expected = carry, {"values": expected_values, "total": torch.stack(output_totals)}
    torch.testing.assert_close(run(values), expected)


def test_scan_template_supplies_only_structure() -> None:
    template = _Values(
        current=torch.empty(9, 7, device="meta", dtype=torch.int64),
        total=torch.empty((), device="meta", dtype=torch.float64),
    )

    def run(values: Tensor) -> tuple[Tensor, _Values]:
        def combine_fn(carry: Tensor, item: Tensor) -> tuple[Tensor, _Values]:
            updated = carry + item
            return updated, _Values(current=updated.clone(), total=updated.sum())

        return torch_scan(combine_fn, torch.zeros(2), values, output_template=template)

    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    carry, output = run(values)

    torch.testing.assert_close(carry, values.sum(0))
    torch.testing.assert_close(output.current, values.cumsum(0))
    torch.testing.assert_close(output.total, values.sum(1).cumsum(0))
    assert template.current.device.type == "meta"
    assert template.current.shape == (9, 7)


def test_scan_empty_output_tree_preserves_its_structure() -> None:
    def run(values: Tensor) -> tuple[Tensor, tuple[()]]:
        return torch_scan(lambda carry, item: (carry + item, ()), torch.zeros(()), values, output_template=())

    carry, output = run(torch.ones(3))

    torch.testing.assert_close(carry, torch.tensor(3.0))
    assert output == ()
