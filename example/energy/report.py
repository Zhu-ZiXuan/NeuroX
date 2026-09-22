"""Aggregate named Reporter data for model energy evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from neurox import ProfileItem, Reporter

from .observer import EnergyObserver


def summarize_profile(
    result: dict[str, ProfileItem],
    observer: EnergyObserver,
    *,
    single_sample_batches: bool = False,
    powered_duration__ns: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Add layer totals and dataset-sample statistics after named PPA preparation.

    Restore convolution leading axes and reduce token/time operations to
    dataset samples. In the single-sample evaluator, every operation in one
    inference belongs to that dataset sample. Hardware is counted once per name.
    Apply powered-window overrides in the collected operation layout before
    dataset-sample aggregation.
    """
    named = {}
    collected_shapes = {}
    for operator in observer.operators.values():
        unit = operator.unit
        if unit is None:
            raise RuntimeError(f"{operator.source} was not prepared")
        name = unit.qualified_name
        named[name] = operator
        collected_shapes[name] = (
            [(math.prod(shape),) for shape in operator.leading_shapes]
            if operator.convolution
            else operator.leading_shapes
        )
    reporter = Reporter(result, powered_duration__ns=powered_duration__ns)
    operators: dict[str, Any] = {}
    model_dynamic: Tensor | None = None
    model_static: Tensor | None = None
    for name, operator in named.items():
        items = [item for key, item in reporter.data.items() if key == name or key.startswith(name + ".")]
        area = _sum_known([item.area__um2 for item in items])
        leakage = _sum_known([item.leakage__uW for item in items])
        dynamic_samples = None
        static_samples = None
        for item in items:
            if item.dynamic_energy__fJ is not None:
                dynamic_samples = _add(
                    dynamic_samples,
                    _dataset_samples(
                        item.dynamic_energy__fJ,
                        collected_shapes[name],
                        operator.leading_shapes,
                        single_sample_batches=single_sample_batches,
                    ),
                )
            if item.static_energy__fJ is not None:
                static_samples = _add(
                    static_samples,
                    _dataset_samples(
                        item.static_energy__fJ,
                        collected_shapes[name],
                        operator.leading_shapes,
                        single_sample_batches=single_sample_batches,
                    ),
                )
        total_samples = _add(dynamic_samples, static_samples)
        model_dynamic = _add(model_dynamic, dynamic_samples)
        model_static = _add(model_static, static_samples)
        operators[name] = {
            "source": operator.source,
            "input_bits": operator.input_bits,
            "weight_bits": operator.weight_bits,
            "input_encoding": operator.input_mode,
            "area__um2": area,
            "leakage__uW": leakage,
            "leading_shapes": operator.leading_shapes,
            "working_duration__ns": _tolist(reporter.data[name].working_duration__ns),
            "powered_duration__ns": _tolist(reporter.data[name].powered_duration__ns),
            "sample_dynamic_energy__fJ": _tolist(dynamic_samples),
            "sample_static_energy__fJ": _tolist(static_samples),
            "sample_total_energy__fJ": _tolist(total_samples),
            "dynamic_energy__fJ": _total(dynamic_samples),
            "static_energy__fJ": _total(static_samples),
            "total_energy__fJ": _total(total_samples),
        }
    model_total = _add(model_dynamic, model_static)
    return {
        "preset": observer.factory.preset,
        "hardware_config": str(observer.factory.config_file),
        "ideal_macro": observer.factory.ideal_macro,
        "analog_dynamic_energy_included": not observer.factory.ideal_macro,
        "static_ppa_scope": "ideal_macro_local_and_digital" if observer.factory.ideal_macro else "physical_units",
        "static_energy_basis": "explicit_powered_windows; recorded_working_duration_elsewhere"
        if powered_duration__ns
        else "recorded_basic_operation_duration; sequential operations; powered_off_when_idle",
        "merge": observer.factory.merge,
        "quantization_defaults": {
            "input_bits": observer.factory.input_bits,
            "weight_bits": observer.factory.weight_bits,
            "floating_operands": "unsupported_without_explicit_model_quantization",
        },
        "digital_cost_assumptions__fJ": {
            "local_accumulator": observer.factory.local_accumulator.energy_per_op__fJ,
            "radix_summator": None
            if observer.factory.radix_summator is None
            else observer.factory.radix_summator.energy_per_op__fJ,
            "tile_accumulator": None
            if observer.factory.tile_accumulator is None
            else observer.factory.tile_accumulator.energy_per_op__fJ,
            "weight_polarity": None
            if observer.factory.polarity_adder is None
            else observer.factory.polarity_adder.energy_per_op__fJ,
        },
        "operator_count": len(operators),
        "operator_calls": sum(len(operator.leading_shapes) for operator in named.values()),
        "area__um2": _sum_known([item.area__um2 for item in reporter.data.values()]),
        "leakage__uW": _sum_known([item.leakage__uW for item in reporter.data.values()]),
        "dynamic_energy__fJ": _total(model_dynamic),
        "static_energy__fJ": _total(model_static),
        "total_energy__fJ": _total(model_total),
        "sample_energy__fJ": {
            "dynamic_energy": _tolist(model_dynamic),
            "static_energy": _tolist(model_static),
            "total_energy": _tolist(model_total),
        },
        "operators": operators,
    }


def _dataset_samples(
    values: Tensor,
    collected_shapes: Sequence[tuple[int, ...]],
    leading_shapes: Sequence[tuple[int, ...]],
    *,
    single_sample_batches: bool,
) -> Tensor:
    parts = values.split([shape[0] for shape in collected_shapes], dim=0)
    if single_sample_batches:
        return torch.stack([part.sum() for part in parts])
    samples = []
    for part, shape in zip(parts, leading_shapes, strict=True):
        aligned = part.reshape(shape)
        dims = tuple(range(1, aligned.ndim))
        samples.append(aligned.sum(dim=dims) if dims else aligned)
    return torch.cat(samples)


def _sum_known(values: Sequence[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _add(first: Tensor | None, second: Tensor | None) -> Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    if first.shape != second.shape:
        raise ValueError("operator results describe different dataset samples")
    return first + second.to(first.device)


def _total(value: Tensor | None) -> float | None:
    return None if value is None else value.sum().item()


def _tolist(value: Tensor | None) -> list[object] | None:
    return None if value is None else list(value.cpu().tolist())
