"""Composable PPA measurement capabilities for CIM-macro validation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.api.profiler import Profiler
from neurox.api.reporter import Reporter
from neurox.primitive.macro.cim import CimMacro

type _Macro = CimMacro


@dataclass(frozen=True)
class VmmProfile:
    """Per-access dynamic energy from one complete batched VMM call."""

    dynamic_by_name_per_access__fJ: dict[str, float]
    vmm_num: int
    access_num: int


def profile_vmm(
    macro: _Macro,
    reporter: Reporter,
    x: Tensor,
    *,
    quantization_mode: int,
    adc_active_bits: int,
) -> VmmProfile:
    """Profile one complete VMM call and normalize it per serialized access.

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

    vmm_num = math.prod(x.shape[:-1])
    access_num = vmm_num * macro.scan_num
    with torch.no_grad(), Profiler() as profiler:
        macro.vec_mat_mul(
            x,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
    dynamic_by_name_per_access__fJ = {
        name: energy__fJ / access_num for name, energy__fJ in reporter.by_name(profiler).items()
    }
    return VmmProfile(
        dynamic_by_name_per_access__fJ=dynamic_by_name_per_access__fJ,
        vmm_num=vmm_num,
        access_num=access_num,
    )


def static_energy_by_name__fJ(
    macro: _Macro,
    reporter: Reporter,
    *,
    measurement_cycle__ns: float,
) -> dict[str, float]:
    """Return mean per-macro static energy over one measurement cycle."""
    if measurement_cycle__ns <= 0.0:
        raise ValueError(f"measurement_cycle__ns must be positive; got {measurement_cycle__ns}")
    return {
        entry.qualified_name: entry.leakage__uW / macro.inst_count * measurement_cycle__ns
        for entry in reporter.static_entries
    }


def area_per_macro__um2(macro: _Macro, reporter: Reporter) -> float:
    """Return mean area per macro instance."""
    area__um2: float = reporter.static.area__um2 / macro.inst_count
    return area__um2


def mean_by_name(rounds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average named scalar rows across equally weighted rounds."""
    if not rounds:
        raise ValueError("at least one round is required")
    names = dict.fromkeys(name for round_data in rounds for name in round_data)
    return {name: statistics.fmean(round_data.get(name, 0.0) for round_data in rounds) for name in names}


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
