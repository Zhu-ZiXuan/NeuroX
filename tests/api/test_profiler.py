"""Profiler keeps named contributions and sample-aligned timing across contexts."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from neurox import Profiler, Reporter, set_profile_leading_rank
from neurox.primitive.digital import Accumulator, AccumulatorConfig, DigitalPolicy


@pytest.mark.parametrize(("on_repeat", "first_factor"), [("sum", 3), ("replace", 2)])
def test_named_contributions_merge_only_within_their_context(on_repeat, first_factor) -> None:
    energy = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    model = nn.ModuleDict({"leaf": _accumulator(1)})
    profiler = Profiler(concat_dim=0, on_repeat=on_repeat)
    profiler.collect_static_data(model)
    with profiler:
        profiler.submit_dynamic_energy(name="leaf", dynamic_energy__fJ=energy, channel="read")
        profiler.submit_dynamic_energy(name="leaf", dynamic_energy__fJ=2 * energy, channel="read")
        profiler.submit_dynamic_energy(name="leaf", dynamic_energy__fJ=energy, channel="idle")
    with profiler:
        profiler.submit_dynamic_energy(name="leaf", dynamic_energy__fJ=3 * energy, channel="read")
        profiler.submit_dynamic_energy(name="leaf", dynamic_energy__fJ=energy, channel="idle")

    result = profiler.result
    assert set(result) == {"leaf", "leaf.read", "leaf.idle"}
    torch.testing.assert_close(
        result["leaf.read"].dynamic_energy__fJ,
        torch.cat((first_factor * energy, 3 * energy), dim=0),
    )
    torch.testing.assert_close(result["leaf.idle"].dynamic_energy__fJ, torch.cat((energy, energy), dim=0))
    assert result["leaf.read"].area__um2 is None
    assert result["leaf.read"].leakage__uW is None


def test_compiled_scalar_submissions_merge_with_singletons_and_concatenate_contexts(device) -> None:
    profiler = Profiler(concat_dim=0)

    @torch.compile(dynamic=False, fullgraph=True)
    def submit(energy, duration):
        profiler.submit_dynamic_energy(name="unit", dynamic_energy__fJ=energy)
        # Normalization must precede merging, so a scalar and a one-position
        # vector contribute to the same observation without a shape mismatch.
        profiler.submit_dynamic_energy(name="unit", dynamic_energy__fJ=2 * energy.reshape(1))
        profiler.submit_latency(name="unit", latency__ns=duration)

    for index, shape in enumerate(((), (1,))):
        energy = torch.full(shape, float(index + 1), device=device, requires_grad=True)
        duration = torch.full(shape, float(index + 2), dtype=torch.float64, device=device, requires_grad=True)
        with profiler:
            submit(energy, duration)
            with torch.no_grad():
                energy.fill_(-1)
                duration.fill_(-1)
        item = profiler.result["unit"]
        expected_energy = 3 * torch.arange(1, index + 2, dtype=torch.float32, device="cpu")
        expected_duration = torch.arange(2, index + 3, dtype=torch.float64, device="cpu")
        torch.testing.assert_close(item.dynamic_energy__fJ, expected_energy)
        torch.testing.assert_close(item.working_duration__ns, expected_duration)
        assert not item.dynamic_energy__fJ.requires_grad
        assert not item.working_duration__ns.requires_grad


def _accumulator(inst_count: int) -> Accumulator:
    return Accumulator(
        config=AccumulatorConfig(
            bit_width=8,
            energy_per_op__fJ=3.0,
            area_per_inst__um2=2.0,
            leakage_per_inst__uW=0.5,
        ),
        policy=DigitalPolicy(),
        inst_shape=(inst_count,),
    )


def test_named_fields_keep_independent_layouts_across_contexts() -> None:
    model = nn.ModuleDict({"unit": _accumulator(1), "unused": _accumulator(2)})
    profiler = Profiler(concat_dim=-1)
    profiler.collect_static_data(model)
    first_energy = torch.arange(6.0).reshape(2, 3)
    second_energy = torch.arange(2.0).reshape(2, 1)
    # A path prefix does not associate parent timing with child energy.
    first_duration = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True).unsqueeze(0)
    second_duration = torch.tensor(5.0, dtype=torch.float64).expand(1, 1)
    for energy, duration in ((first_energy, first_duration), (second_energy, second_duration)):
        with profiler:
            profiler.submit_dynamic_energy(name="unit", channel="read", dynamic_energy__fJ=energy)
            profiler.submit_latency(name="unit", latency__ns=duration)

    result = profiler.result
    torch.testing.assert_close(
        result["unit.read"].dynamic_energy__fJ, torch.cat((first_energy, second_energy), dim=1), check_dtype=False
    )
    durations = result["unit"].working_duration__ns
    assert durations is not None
    torch.testing.assert_close(durations, torch.cat((first_duration.detach(), second_duration), dim=1))
    assert not durations.requires_grad
    assert durations.device == torch.device("cpu")
    assert result["unit.read"].working_duration__ns is None
    assert result["unused"].working_duration__ns is None
    assert result["unused"].area__um2 == 4.0


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_compiled_mixed_device_submissions_are_synchronized_for_reporting(
    device: torch.device, compute_dtype: torch.dtype
) -> None:
    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(nn.ModuleDict({"unit": _accumulator(1)}))

    @torch.compile(dynamic=False, fullgraph=True)
    def emit(x):
        constant = torch.full((x.shape[0],), 3.0, dtype=torch.float32)
        dynamic = x.square().sum(dim=-1)
        duration = torch.full((x.shape[0],), 2.0, dtype=torch.float64)
        profiler.submit_dynamic_energy(name="unit", dynamic_energy__fJ=constant)
        profiler.submit_dynamic_energy(name="unit", dynamic_energy__fJ=dynamic)
        profiler.submit_latency(name="unit", latency__ns=duration)
        return constant, dynamic, duration

    expected_parts = []
    for batch_size in (2, 3):
        x = torch.arange(batch_size * 4, device=device, dtype=compute_dtype).reshape(batch_size, 4)
        with profiler:
            constant, dynamic, duration = emit(x)
            assert constant.device == duration.device == torch.get_default_device()
            assert dynamic.device == device
        expected_parts.append(x.square().sum(dim=-1).cpu() + 3.0)

    result = profiler.result
    assert result["unit"].dynamic_energy__fJ.device.type == "cpu"
    assert result["unit"].working_duration__ns.device.type == "cpu"
    expected_energy = torch.cat(expected_parts)
    torch.testing.assert_close(result["unit"].dynamic_energy__fJ, expected_energy, check_dtype=False)
    total = Reporter(result).breakdown("total_energy")["unit"]
    torch.testing.assert_close(total, expected_energy.double() + 1.0, check_dtype=False)


@pytest.mark.parametrize("compile_model", [False, True])
def test_compiled_emitters_resolve_copied_and_restamped_names(device, compile_model) -> None:
    template = _accumulator(1)
    model = nn.ModuleList([deepcopy(template) for _ in range(12)]).to(device)
    set_profile_leading_rank(model, 1)

    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=True)
    def emit(module, value):
        module._record_dynamic_energy(value, channel="read")

    def execute(modules, values):
        for module, value in zip(modules, values, strict=True):
            emit(module, value)

    if compile_model:
        execute = torch.compile(execute, dynamic=False, fullgraph=True)

    profiler = Profiler(concat_dim=0)
    profiler.collect_static_data(model)
    expected = {str(i): [] for i in range(len(model))}
    for batch in range(2):
        order = list(range(len(model))) if batch == 0 else list(reversed(range(len(model))))
        values = [torch.full((2,), float(i + batch), device=device) for i in order]
        for i, value in zip(order, values, strict=True):
            expected[str(i)].append(value)
        with profiler:
            execute([model[i] for i in order], values)

    result = profiler.result
    for name, parts in expected.items():
        torch.testing.assert_close(result[f"{name}.read"].dynamic_energy__fJ, torch.cat(parts).cpu(), check_dtype=False)

    # Reuse the same compiled boundary and module order after re-stamping.
    renamed = Profiler(concat_dim=0)
    renamed.collect_static_data(nn.ModuleDict({"renamed": model}))
    with renamed:
        execute([model[i] for i in order], values)
    result = renamed.result
    assert set(result) == {f"renamed.{i}{suffix}" for i in range(len(model)) for suffix in ("", ".read")}
    for i, value in zip(order, values, strict=True):
        torch.testing.assert_close(result[f"renamed.{i}.read"].dynamic_energy__fJ, value.cpu(), check_dtype=False)
