"""PyTorch boundary adapters preserve callable types and structured loop states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import pytest
import torch
from torch import Tensor
from torch._dynamo.exc import UncapturedHigherOrderOpError

import neurox.common.torch_compat as torch_compat
from neurox.common.pytree_dataclass_mixin import PyTreeDataClassMixin
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin
from neurox.common.torch_compat import (
    torch_assert_async,
    torch_cond,
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


class _SymbolicState(NamedTuple):
    step: int
    values: Tensor


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


@pytest.mark.parametrize("compiled", [False, True])
def test_cond_preserves_registered_optional_output_and_runtime_selection(device: torch.device, compiled: bool) -> None:
    def run(pred: Tensor, state: _ScanCarry) -> _ScanCarry:
        def true_fn(operand: _ScanCarry) -> _ScanCarry:
            return _ScanCarry(total=operand.total + 1, steps=operand.steps + 1)

        def false_fn(operand: _ScanCarry) -> _ScanCarry:
            return _ScanCarry(total=operand.total - 1, steps=operand.steps + 2)

        return torch_cond(pred, true_fn, false_fn, state, output_template=state)

    execute = torch.compile(run, fullgraph=True) if compiled else run
    initial = _ScanCarry(total=torch.ones(3, device=device), steps=torch.zeros((), device=device))
    for selected in (True, False):
        result = execute(torch.tensor(selected, device=device), initial)
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


def _solve_symbolic_shape(values: Tensor) -> _SymbolicState:
    state = _SymbolicState(step=values.shape[0], values=values)

    def cond_fn(current: _SymbolicState) -> bool:
        return current.step < current.values.shape[0] + 2

    def body_fn(current: _SymbolicState) -> _SymbolicState:
        return _SymbolicState(step=current.step + 1, values=current.values + 1)

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


def test_caller_fullgraph_compile_preserves_semantic_result() -> None:
    values = torch.tensor([1.0, 0.125])

    expected = _solve(values)
    compiled = torch.compile(_solve, backend="eager", fullgraph=True, dynamic=False)
    actual = compiled(values)

    assert isinstance(actual, _State)
    assert isinstance(actual.iteration, _Iteration)
    assert isinstance(actual.values, _Values)
    assert torch.equal(actual.iteration.step, expected.iteration.step)
    assert torch.equal(actual.iteration.active, expected.iteration.active)
    torch.testing.assert_close(actual.values.current, expected.values.current)
    torch.testing.assert_close(actual.values.total, expected.values.total)


def test_registered_tensor_dataclass_works_direct_and_caller_compiled() -> None:
    values = torch.tensor([1.0, 0.125])

    direct = _solve_dataclass(values)
    compiled = torch.compile(_solve_dataclass, backend="eager", fullgraph=True, dynamic=False)
    captured = compiled(values)

    assert type(direct) is _DataClassState
    assert type(captured) is _DataClassState
    assert int(captured.step) == 3
    assert not torch.any(captured.active)
    torch.testing.assert_close(captured.values, torch.tensor([0.125, 0.125]))
    assert torch.equal(captured.step, direct.step)
    assert torch.equal(captured.active, direct.active)
    torch.testing.assert_close(captured.values, direct.values)


def test_python_integer_and_boolean_states_follow_native_contract() -> None:
    step = torch_while_loop(lambda current: current < 3, lambda current: current + 1, 0)
    active = torch_while_loop(lambda current: current, lambda _current: False, True)

    assert step == 3
    assert active is False


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


def test_symbolic_integer_state_works_with_dynamic_shapes() -> None:
    compiled = torch.compile(_solve_symbolic_shape, backend="eager", fullgraph=True, dynamic=True)
    results = [compiled(torch.ones(size)) for size in (3, 5)]

    for size, result in zip((3, 5), results, strict=True):
        assert result.step == size + 2
        torch.testing.assert_close(result.values, torch.full((size,), 3.0))


def test_async_assert_is_captured_by_a_fullgraph_caller() -> None:
    def require_finite(values: Tensor) -> Tensor:
        torch_assert_async(values.isfinite().all(), "values must be finite")
        return values + 1.0

    compiled = torch.compile(require_finite, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
    with pytest.raises(RuntimeError, match="values must be finite"):
        compiled(torch.tensor([0.0, torch.nan]))


@pytest.mark.parametrize("unsupported", [0.0, "invalid"])
def test_unsupported_leaf_is_rejected_before_loop_capture(unsupported: object) -> None:
    with pytest.raises(TypeError, match="carried_state must contain only Tensor, int, or SymInt leaves"):
        torch_while_loop(
            lambda _state: torch.tensor(False),
            lambda state: state,
            {"value": torch.tensor(1.0), "unsupported": unsupported},
        )


def test_body_must_preserve_state_structure() -> None:
    initial = _initial_state(torch.tensor([1.0]))

    def wrong_body(current: _State) -> _State:
        return cast(_State, (current.iteration, current.values))

    with pytest.raises(UncapturedHigherOrderOpError) as exc_info:
        torch_while_loop(lambda current: current.iteration.step < 1, wrong_body, initial)

    message = str(exc_info.value)
    assert "body output must preserve the carried-state PyTree structure" in message
    assert "expected TreeSpec" in message
    assert "got TreeSpec" in message


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
def test_scan_uses_leading_axis_for_different_rank_leaves_direct_and_fullgraph(length: int, reverse: bool) -> None:
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

    direct = _scan_dataclass(values, reverse)
    compiled = torch.compile(_scan_dataclass, backend="eager", fullgraph=True)
    captured = compiled(values, reverse)

    for final_carry, output in (direct, captured):
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
    compiled = torch.compile(run, backend="eager", fullgraph=True)

    for actual in (run(values), compiled(values)):
        torch.testing.assert_close(actual, expected)


def test_scan_calls_native_with_fixed_options(monkeypatch: pytest.MonkeyPatch) -> None:
    native_scan = torch_compat.scan
    calls: list[tuple[int, bool]] = []

    def recording_scan(combine_fn, init, xs, *, dim: int, reverse: bool):
        calls.append((dim, reverse))
        return native_scan(combine_fn, init, xs, dim=dim, reverse=reverse)

    monkeypatch.setattr(torch_compat, "scan", recording_scan)
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch_scan(
        lambda carry, item: (carry + item, (carry + item).clone()),
        torch.zeros(2),
        values,
        dim=-1,
        reverse=True,
        output_template=torch.empty(()),
    )

    assert calls == [(0, False)]


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
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    carry, output = compiled(values)

    torch.testing.assert_close(carry, values.sum(0))
    torch.testing.assert_close(output.current, values.cumsum(0))
    torch.testing.assert_close(output.total, values.sum(1).cumsum(0))
    assert template.current.device.type == "meta"
    assert template.current.shape == (9, 7)


def test_scan_dynamic_lengths_remain_a_higher_order_loop() -> None:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph: torch.fx.GraphModule, _inputs: list[object]):
        graphs.append(graph)
        return graph.forward

    def run(values: Tensor) -> tuple[_ScanCarry, _ScanOutput]:
        return _scan_dataclass(values)

    compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
    for length in (4, 7):
        values = torch.ones(length, 2, 3)
        carry, output = compiled(values)
        expected = 2.0 * (1.0 - 0.5 ** torch.arange(1, length + 1))
        torch.testing.assert_close(output.values.total, expected[:, None, None].expand(length, 2, 3))
        torch.testing.assert_close(carry.total, output.values.total[-1])

    assert len(graphs) == 1
    assert sum(node.target is torch.ops.higher_order.scan for node in graphs[0].graph.nodes) == 1


@pytest.mark.parametrize("compiled", [False, True], ids=["direct", "inductor"])
def test_scan_forward_elimination_and_reverse_substitution(compiled: bool) -> None:
    def solve(rhs: Tensor) -> Tensor:
        def eliminate(carry: tuple[Tensor, Tensor], item: tuple[Tensor, Tensor]):
            previous_upper, previous_rhs = carry
            diagonal, current_rhs = item
            denominator = diagonal - previous_upper
            upper = 1.0 / denominator
            reduced_rhs = (current_rhs - previous_rhs) / denominator
            return (upper, reduced_rhs), (upper.clone(), reduced_rhs.clone())

        init = (rhs.new_zeros(()), torch.zeros_like(rhs[..., 0]))
        _, reduced = torch_scan(
            eliminate, init, (rhs.new_full((rhs.shape[-1],), 4.0), rhs.movedim(-1, 0)), output_template=init
        )

        def substitute(next_value: Tensor, item: tuple[Tensor, Tensor]):
            upper, reduced_rhs = item
            value = reduced_rhs - upper * next_value
            return value, value.clone()

        initial_value = torch.zeros_like(rhs[..., 0])
        _, solution = torch_scan(substitute, initial_value, reduced, output_template=initial_value, reverse=True)
        return solution.movedim(0, -1)

    rhs = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    matrix = 4.0 * torch.eye(5, dtype=rhs.dtype)
    matrix += torch.diag(torch.ones(4, dtype=rhs.dtype), diagonal=1)
    matrix += torch.diag(torch.ones(4, dtype=rhs.dtype), diagonal=-1)
    expected = torch.linalg.solve(matrix, rhs.T).T

    runner = torch.compile(solve, backend="inductor", fullgraph=True) if compiled else solve
    actual = runner(rhs)

    torch.testing.assert_close(actual, expected)


def test_scan_empty_output_tree_preserves_its_structure() -> None:
    def run(values: Tensor) -> tuple[Tensor, tuple[()]]:
        return torch_scan(lambda carry, item: (carry + item, ()), torch.zeros(()), values, output_template=())

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    carry, output = compiled(torch.ones(3))

    torch.testing.assert_close(carry, torch.tensor(3.0))
    assert output == ()


@pytest.mark.parametrize("name", ["init", "xs", "output_template"])
@pytest.mark.parametrize("unsupported", [1, True, 0.5, "invalid"])
def test_scan_rejects_non_tensor_leaves(name: str, unsupported: object) -> None:
    init = {"value": unsupported if name == "init" else torch.zeros(())}
    xs = {"value": unsupported if name == "xs" else torch.ones(3)}
    template = {"value": unsupported if name == "output_template" else torch.empty(())}

    with pytest.raises(TypeError, match=f"{name} must contain only Tensor leaves"):
        torch_scan(lambda carry, item: (carry, item), init, xs, output_template=template)


@pytest.mark.parametrize("name", ["carry output", "scan output"])
def test_scan_rejects_non_tensor_output_leaves(name: str) -> None:
    def combine_fn(carry: Tensor, item: Tensor) -> tuple[object, object]:
        return (1, item.clone()) if name == "carry output" else (carry + item, 1)

    with (
        pytest.raises(UncapturedHigherOrderOpError, match=f"{name} must contain only Tensor leaves"),
    ):
        torch_scan(combine_fn, torch.zeros(()), torch.ones(3), output_template=torch.empty(()))


def test_scan_rejects_changed_carry_structure() -> None:
    with (
        pytest.raises(UncapturedHigherOrderOpError, match="carry output must preserve the init PyTree structure"),
    ):
        torch_scan(
            lambda carry, item: ((carry + item,), item.clone()),
            torch.zeros(()),
            torch.ones(3),
            output_template=torch.empty(()),
        )


@pytest.mark.parametrize("length", [1, 3])
def test_scan_output_must_match_template_structure(length: int) -> None:
    with (
        pytest.raises(
            UncapturedHigherOrderOpError, match="scan output must match the output_template PyTree structure"
        ),
    ):
        torch_scan(
            lambda carry, item: (carry + item, (item.clone(),)),
            torch.zeros(()),
            torch.ones(length),
            output_template=torch.empty(()),
        )


@pytest.mark.parametrize("xs", [torch.empty(0), torch.zeros(())])
def test_scan_rejects_inputs_without_a_positive_scan_dimension(xs: Tensor) -> None:
    with pytest.raises(ValueError, match=r"scan dimension must have positive length|must have dimension"):
        torch_scan(
            lambda carry, item: (carry + item, item.clone()), torch.zeros(()), xs, output_template=torch.empty(())
        )


def test_scan_rejects_input_leaves_with_different_scan_lengths() -> None:
    xs = {"short": torch.ones(3), "long": torch.ones(4)}

    with pytest.raises(ValueError, match="same scan dimension length"):
        torch_scan(
            lambda carry, item: (carry + item["short"] + item["long"], carry.clone()),
            torch.zeros(()),
            xs,
            output_template=torch.empty(()),
        )


@pytest.mark.parametrize("empty", ["init", "xs"])
def test_scan_rejects_empty_carry_or_input_trees(empty: str) -> None:
    with pytest.raises(ValueError, match="init and xs must each contain at least one Tensor leaf"):
        torch_scan(
            lambda carry, item: (carry, item),
            () if empty == "init" else torch.zeros(()),
            () if empty == "xs" else torch.ones(3),
            output_template=torch.empty(()),
        )
