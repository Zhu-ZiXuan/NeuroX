"""Diagnostic collection preserves typed histories and solver contracts."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox import Profiler, TraceCollector
from neurox.execution.compilation import compile_recording
from neurox.execution.loop import run_scan
from neurox.execution.solving import SolvingTrace


class _InnerTrace(SolvingTrace):
    residual: Tensor
    limited: Tensor


class _OuterTrace(SolvingTrace):
    inner: _InnerTrace
    iteration: Tensor
    absent: _InnerTrace | None


@pytest.mark.parametrize("compiled", [False, True])
def test_completed_trace_preserves_fields_dtypes_and_invalid_values(device, compiled):
    def produce(value):
        trace = _OuterTrace(
            inner=_InnerTrace(residual=value, limited=torch.isnan(value)),
            iteration=torch.tensor([0, -1], dtype=torch.int64, device=device),
            absent=None,
        )
        trace.record(source="solver")
        return trace

    function = compile_recording(produce, fullgraph=True) if compiled else produce
    values = torch.tensor([1.0, torch.nan], dtype=torch.float64, device=device)
    with TraceCollector() as collector:
        result = function(values)
    assert len(collector.result) == 1
    record = collector.result[0]
    assert record.source == "solver"
    assert set(record.values) == {"inner.residual", "inner.limited", "iteration"}
    torch.testing.assert_close(record.values["inner.residual"], result.inner.residual.cpu(), equal_nan=True)
    torch.testing.assert_close(record.values["inner.limited"], result.inner.limited.cpu())
    torch.testing.assert_close(record.values["iteration"], result.iteration.cpu())
    values.fill_(7)
    assert torch.isnan(record.values["inner.residual"][1])


def test_nested_trace_pool_keeps_both_history_axes_and_coexists_with_energy(device):
    profiler = Profiler(concat_dim=0)
    collector = TraceCollector()

    @compile_recording(fullgraph=True)
    def execute(values):
        def outer(state, row):
            def inner(counter, value):
                collector.submit({"residual": value, "valid": value >= 0}, source="node")
                profiler.submit_dynamic_energy(value.square(), name="array", leading_rank=0)
                return counter + 1, value + 2

            _, result = run_scan(
                init_state=torch.zeros((), device=device),
                xs=row,
                body_fn=inner,
                output_template=row[0],
                record_dim=-1,
            )
            return state + 1, result

        return run_scan(
            init_state=torch.zeros((), device=device),
            xs=values,
            body_fn=outer,
            output_template=values[0],
            record_dim=-1,
        )

    values = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 4, 2)
    with profiler, collector, torch.no_grad():
        _, result = execute(values)
    torch.testing.assert_close(result, values + 2)
    torch.testing.assert_close(collector.result[0].values["residual"], values.permute(2, 1, 0).cpu())
    torch.testing.assert_close(profiler.result["array"].dynamic_energy__fJ, values.square().sum().cpu().reshape(1))


def test_failed_context_discards_trace_and_releases_collection(device):
    collector = TraceCollector()

    def abort():
        with collector:
            collector.submit({"value": torch.ones(2, device=device)}, source="one")
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        abort()
    assert not collector.result
    assert not TraceCollector.active()
    with collector:
        collector.submit({"value": torch.zeros(2, device=device)}, source="two")
    assert [record.source for record in collector.result] == ["two"]
