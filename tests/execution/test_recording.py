"""Scoped output records preserve positions, snapshots, and submission ownership."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from neurox import Profiler
from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin
from neurox.common.recorder import RecorderBase
from neurox.common.torch_compat import torch_scan
from neurox.execution.chunking import run_chunked
from neurox.execution.compilation import compile_recording
from neurox.execution.loop import run_map, run_scan


class _Records(RecorderBase[Tensor, Tensor, list[Tensor]]):
    @property
    def result(self):
        return self._history_records

    @RecorderBase.submission
    def submit(self, value: Tensor) -> None:
        self._submit_record(self._export_tensor(value))

    def _merge_records(self, records: Sequence[Tensor]) -> Sequence[Tensor]:
        return records


class _Values(TensorDataClassMixin, PyTreeDataClassMixin):
    values: Tensor


@pytest.mark.parametrize("compiled", [False, True])
def test_scan_and_map_submit_automatically_with_native_record_axes(device, compiled):
    recorder = _Records()

    def leaf(x):
        recorder.submit(x.square())
        return x + 3

    def execute(xs):
        def body(state, x):
            ys = run_map(xs=x, body_fn=leaf, output_template=x[0])
            return state + 1, ys

        return run_scan(
            init_state=torch.zeros((), device=device),
            xs=xs,
            body_fn=body,
            output_template=xs[0],
            record_dim=-1,
        )

    function = compile_recording(execute, fullgraph=True, backend="eager") if compiled else execute
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3)
    with recorder, torch.no_grad():
        state, result = function(x)
    torch.testing.assert_close(result, x + 3)
    torch.testing.assert_close(state, torch.tensor(4.0, device=device))
    assert len(recorder.result) == 1
    torch.testing.assert_close(recorder.result[0], x.square().movedim(0, -1).cpu())


@pytest.mark.parametrize("chunk_size", [0, 5, 7])
def test_chunk_records_restore_observation_axes_before_reduction(device, chunk_size):
    profiler = Profiler(concat_dim=0)
    values = torch.arange(24, dtype=torch.float64, device=device).reshape(2, 3, 4)

    def body(inputs):
        profiler.submit_dynamic_energy(inputs.values.square(), name="unit", leading_rank=2)
        return _Values(values=inputs.values + 1)

    with profiler, torch.no_grad():
        result = run_chunked(
            expected_chunk_size=chunk_size,
            leading_shape=(2, 3, 4),
            device=device,
            operands=_Values(values=values),
            output_template=_Values(values=torch.empty(0)),
            body_fn=body,
        )
    torch.testing.assert_close(result.values, values + 1)
    torch.testing.assert_close(profiler.result["unit"].dynamic_energy__fJ, values.square().sum(-1).cpu())


def test_raw_scan_is_not_enhanced(device):
    recorder = _Records()

    @compile_recording(fullgraph=True, backend="eager")
    def execute(xs):
        def body(state, value):
            recorder.submit(value)
            return state + 1, value.clone()

        return torch_scan(body, torch.zeros((), device=device), xs, output_template=xs[0])

    with recorder, torch.no_grad(), pytest.raises(Exception, match="unsupported boundary"):
        execute(torch.ones(3, 2, device=device))
    assert not recorder.result


def test_observation_snapshot_precedes_later_mutation(device):
    recorder = _Records()

    @compile_recording(fullgraph=True, backend="eager")
    def execute(xs):
        def body(state, value):
            copy = value.clone()
            recorder.submit(copy)
            copy.add_(100)
            return state + 1, copy

        return run_scan(init_state=torch.zeros((), device=device), xs=xs, body_fn=body, output_template=xs[0])

    values = torch.arange(6, device=device, dtype=torch.float32).reshape(3, 2)
    with recorder, torch.no_grad():
        _, result = execute(values)
    torch.testing.assert_close(result, values + 100)
    torch.testing.assert_close(recorder.result[0], values.cpu())


@pytest.mark.parametrize("reverse", [False, True])
def test_record_axes_follow_reverse_scan_and_nonleading_input_axis(device, reverse):
    recorder = _Records()

    @compile_recording(fullgraph=True)
    def execute(xs):
        def body(state, value):
            recorder.submit(value + 7)
            return state + 1, value + 2

        return run_scan(
            init_state=torch.zeros((), device=device),
            xs=xs,
            dim=1,
            reverse=reverse,
            body_fn=body,
            output_template=xs[:, 0],
        )

    values = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 4, 3)
    with recorder, torch.no_grad():
        _, result = execute(values)
    torch.testing.assert_close(result, values + 2)
    torch.testing.assert_close(recorder.result[0], (values + 7).cpu())


@pytest.mark.parametrize("control", ["while", "cond"])
def test_unsupported_control_flow_cannot_hide_records(device, control):
    recorder = _Records()

    @compile_recording(fullgraph=True, backend="eager")
    def execute(value):
        def body(x):
            recorder.submit(x)
            return x + 1

        if control == "cond":
            return torch.cond(value.sum() > 0, body, lambda x: x - 1, (value,))
        return torch.while_loop(lambda x: x.sum() < 5, lambda x: (body(x),), (value,))

    with recorder, torch.no_grad(), pytest.raises(Exception, match="unsupported boundary"):
        execute(torch.ones(2, device=device))
    assert not recorder.result


def test_numerical_while_with_scans_has_no_hidden_energy(device):
    recorder = _Records()

    @compile_recording(fullgraph=True)
    def execute(values):
        def iteration(step, value):
            def scan_body(total, x):
                return total + x, x + 1

            _, value = run_scan(
                init_state=torch.zeros_like(value[0]),
                xs=value,
                body_fn=scan_body,
                output_template=value[0],
            )
            return step + 1, value

        _, result = torch.while_loop(
            lambda step, value: step < 2,
            iteration,
            (torch.zeros((), dtype=torch.int64, device=device), values.clone()),
        )
        recorder.submit(result)
        return result

    values = torch.ones(3, 2, device=device)
    with recorder, torch.no_grad():
        result = execute(values)
    torch.testing.assert_close(result, values + 2)
    assert len(recorder.result) == 1
    torch.testing.assert_close(recorder.result[0], (values + 2).cpu())


def test_cached_calls_route_records_to_the_current_context(device):
    @compile_recording(fullgraph=True)
    def execute(values):
        def body(state, x):
            recorder = _Records.current()
            if recorder is not None:
                recorder.submit(x.square())
            return state + 1, x + 1

        return run_scan(
            init_state=torch.zeros((), device=device),
            xs=values,
            body_fn=body,
            output_template=values[0],
        )

    values = torch.arange(6, dtype=torch.float32, device=device).reshape(3, 2)
    for offset in (0, 10):
        with _Records() as recorder, torch.no_grad():
            execute(values + offset)
            execute(values + offset + 1)
        assert len(recorder.result) == 2
        torch.testing.assert_close(recorder.result[0], (values + offset).square().cpu())
        torch.testing.assert_close(recorder.result[1], (values + offset + 1).square().cpu())
    with torch.no_grad():
        _, result = execute(values)
    torch.testing.assert_close(result, values + 1)


@pytest.mark.parametrize("variant", ["without_output", "without_carry", "without_inputs", "map"])
def test_standalone_loop_variants_collect_without_caller_plumbing(device, variant):
    from neurox.execution.loop import run_scan_without_carry, run_scan_without_inputs, run_scan_without_output

    recorder = _Records()
    values = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)

    def leaf(value):
        recorder.submit(value)
        return value * 2

    def accumulate(state, value):
        recorder.submit(value)
        return state + value

    def advance(state):
        next_state = state + 1
        recorder.submit(next_state)
        return next_state, next_state.clone()

    with recorder, torch.no_grad():
        if variant == "without_output":
            result = run_scan_without_output(
                init_state=torch.zeros(2, device=device),
                xs=values,
                body_fn=accumulate,
                device=device,
            )
            torch.testing.assert_close(result, values.sum(0))
        elif variant == "without_carry":
            result = run_scan_without_carry(xs=values, body_fn=leaf, output_template=values[0], device=device)
            torch.testing.assert_close(result, values * 2)
        elif variant == "without_inputs":
            result, _ = run_scan_without_inputs(
                init_state=torch.zeros(2, device=device),
                length=4,
                body_fn=advance,
                output_template=values[0],
                device=device,
            )
            torch.testing.assert_close(result, torch.full((2,), 4.0, device=device))
        else:
            result = run_map(xs=values, body_fn=leaf, output_template=values[0])
            torch.testing.assert_close(result, values * 2)
    assert len(recorder.result) == 1
    expected = (
        torch.arange(1, 5, dtype=torch.float32).unsqueeze(-1).expand(-1, 2)
        if variant == "without_inputs"
        else values.cpu()
    )
    torch.testing.assert_close(recorder.result[0], expected)
