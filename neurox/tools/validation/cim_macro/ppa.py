"""Raw profiler collection and accounting helpers for CIM-macro validation."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from neurox.api.profiler import ProfileItem, Profiler
from neurox.api.reporter import Reporter
from neurox.primitive.macro.cim import CimMacro


def profile_vmm(
    macro: CimMacro,
    profiler: Profiler,
    x: Tensor,
    *,
    quantization_mode: int,
    adc_active_bits: int,
) -> None:
    """Collect one full VMM batch and its modeled working duration.

    Configure the macro's retained observation axes before collection. The
    recorded duration comes from the macro's latency query and covers one
    complete VMM at each sample position. Experimental powered intervals are
    supplied separately to reporting.

    Args:
        x: Input samples with the macro-instance axes immediately before the
            input axis.
            Shape: `[..., *inst_shape, input]`.
    """
    if x.ndim == 0 or x.shape[-1] != macro.input_num:
        raise ValueError(f"x must end in input_num ({macro.input_num}); got shape {tuple(x.shape)}")
    inst_rank = len(macro.inst_shape)
    input_inst_shape = tuple(x.shape[-(inst_rank + 1) : -1]) if inst_rank else ()
    if input_inst_shape != macro.inst_shape:
        raise ValueError(f"x instance axes must equal macro.inst_shape {macro.inst_shape}; got {input_inst_shape}")

    with torch.no_grad(), profiler:
        macro.vec_mat_mul(x, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        duration__ns = macro.latency__ns(adc_active_bits=adc_active_bits)
        profiler.submit_latency(
            x.new_tensor(duration__ns, dtype=torch.float64).expand(x.shape[:-1]), name=macro.qualified_name
        )


def dynamic_energy_by_round__fJ(
    result: Mapping[str, ProfileItem], *, scan_num: int, n_x: int, groups: Mapping[str, str] | None = None
) -> list[dict[str, float]]:
    """Average VMM samples within each experimental repeat, then convert to scans.

    The campaign supplies each repeat's input count `n_x`. Reporter preserves
    every input and weight sample; paper grouping and statistics happen here.
    """
    reporter = Reporter(result)
    return _rounds(reporter.breakdown("dynamic_energy"), n_x=n_x, divisor=scan_num, groups=groups)


def static_energy_by_round__fJ(
    result: Mapping[str, ProfileItem],
    *,
    scan_num: int,
    n_x: int,
    powered_duration__ns: Tensor,
    groups: Mapping[str, str] | None = None,
) -> list[dict[str, float]]:
    """Summarize per-macro static costs using the campaign's powered intervals."""
    reporter = Reporter(result, powered_duration__ns={"": powered_duration__ns})
    return _rounds(reporter.breakdown("static_energy"), n_x=n_x, divisor=scan_num, groups=groups)


def area_per_macro__um2(result: Mapping[str, ProfileItem]) -> float:
    """Sum supplied local areas for one macro instance."""
    reporter = Reporter(result)
    values = [area for area in reporter.breakdown("area").values() if area is not None]
    if not values:
        raise ValueError("profile result contains no hardware area data")
    return sum(values)


def _rounds(
    values: Mapping[str, Tensor | None],
    *,
    n_x: int,
    divisor: int,
    groups: Mapping[str, str] | None,
) -> list[dict[str, float]]:
    rounds: list[dict[str, float]] = []
    for name, samples in values.items():
        if samples is None:
            continue
        label = name if groups is None else groups.get(name, "other")
        for index, repeat in enumerate(samples.split(n_x, dim=0)):
            if index == len(rounds):
                rounds.append({})
            row = rounds[index]
            row[label] = row.get(label, 0.0) + repeat.mean().item() / divisor
    return rounds


def mean_by_name(rounds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average named scalar rows across equally weighted, complete rounds."""
    if not rounds:
        raise ValueError("at least one round is required")
    names = rounds[0].keys()
    if any(round_data.keys() != names for round_data in rounds[1:]):
        raise ValueError("rounds must contain the same energy names")
    return {name: statistics.fmean(round_data[name] for round_data in rounds) for name in names}


def render_run_summary(
    *,
    device: torch.device,
    n_w: int,
    n_x: int,
    repeat: int,
    scan_num: int,
    seed: int,
) -> str:
    """Render the common workload and physical-access summary."""
    vmm_num = repeat * n_w * n_x
    access_num = vmm_num * scan_num
    return (
        f"Device {device}; {n_w} weight draws x {n_x} inputs x {repeat} rounds = "
        f"{vmm_num} VMM samples per operating point; {scan_num} serialized accesses per VMM, "
        f"{access_num} physical accesses per operating point; seed {seed}."
    )
