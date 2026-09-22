"""Profiler keeps named contributions and sample-aligned timing across contexts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox import Profiler, set_profile_leading_rank
from neurox.primitive.digital import Accumulator, AccumulatorConfig, DigitalPolicy


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize(("on_repeat", "first_factor"), [("sum", 3), ("replace", 2)])
def test_named_contributions_merge_only_within_their_context(rank, on_repeat, first_factor) -> None:
    energy = torch.arange(2 * 5 * 3 * 7, dtype=torch.float64).reshape(2, 5, 3, 7)
    model = nn.ModuleDict({"leaf": _accumulator(1)})
    set_profile_leading_rank(model, rank)
    profiler = Profiler(concat_dim=0, on_repeat=on_repeat)
    profiler.collect_static_data(model)
    with profiler:
        model["leaf"]._record_dynamic_energy(energy, channel="read")
        model["leaf"]._record_dynamic_energy(2 * energy, channel="read")
        model["leaf"]._record_dynamic_energy(energy, channel="idle")
    with profiler:
        model["leaf"]._record_dynamic_energy(3 * energy, channel="read")
        model["leaf"]._record_dynamic_energy(energy, channel="idle")

    expected = energy.reshape(*energy.shape[:rank], -1).sum(dim=-1)
    result = profiler.result
    assert set(result) == {"leaf", "leaf.read", "leaf.idle"}
    torch.testing.assert_close(
        result["leaf.read"].dynamic_energy__fJ, torch.cat((first_factor * expected, 3 * expected), dim=0)
    )
    torch.testing.assert_close(result["leaf.idle"].dynamic_energy__fJ, torch.cat((expected, expected), dim=0))
    assert result["leaf.read"].area__um2 is None
    assert result["leaf.read"].leakage__uW is None


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
    profiler = Profiler(concat_dim=-1, sync_device=torch.device("cpu"))
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
    torch.testing.assert_close(result["unit.read"].dynamic_energy__fJ, torch.cat((first_energy, second_energy), dim=1))
    durations = result["unit"].working_duration__ns
    assert durations is not None
    torch.testing.assert_close(durations, torch.cat((first_duration.detach(), second_duration), dim=1))
    assert not durations.requires_grad
    assert durations.device == torch.device("cpu")
    assert result["unit.read"].working_duration__ns is None
    assert result["unused"].working_duration__ns is None
    assert result["unused"].area__um2 == 4.0
