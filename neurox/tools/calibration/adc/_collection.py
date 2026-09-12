"""Collect paired ADC-input and exact-ideal observations from a CIM macro."""

from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
from torch import Tensor

from neurox.common.module import PolicyBase
from neurox.primitive.analog import AdcProber
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy, IdealCimMacro
from neurox.tools.calibration.cim_macro.construction import build_ideal_twin, build_physical_macro

from ._stimulus import TargetStimulusSampler, feasible_ideal_values, sample_sparse_inputs, sample_values
from .config import AdcProbeToolConfig
from .data import AdcProbeData

logger = logging.getLogger(__name__)


def _integer_values(value_range: tuple[int, int]) -> tuple[int, ...]:
    return tuple(range(value_range[0], value_range[1] + 1))


def _require_all_policy_toggles_off(policy: PolicyBase) -> None:
    enabled: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, bool):
            if value:
                enabled.append(path)
            return
        if not is_dataclass(value):
            return
        for field in fields(value):
            visit(getattr(value, field.name), f"{path}.{field.name}")

    visit(policy, "policy")
    if enabled:
        raise ValueError(f"require: every nonideality policy toggle is off; enabled: {', '.join(enabled)}")


def _ideal_value_range(
    weight_range: tuple[int, int],
    input_range: tuple[int, int],
    max_active_num: int,
) -> tuple[int, int]:
    products = tuple(weight * input_ for weight in weight_range for input_ in input_range)
    return max_active_num * min(0, *products), max_active_num * max(0, *products)


def _run_paired(
    physical: CimMacro[CimMacroConfig, CimMacroPolicy],
    ideal: IdealCimMacro,
    x: Tensor,
    *,
    quantization_mode: int,
) -> tuple[str, Tensor, Tensor]:
    device = next(physical.buffers()).device
    x = x.to(device)
    inst_rank = len(physical.inst_shape)
    prefix_shape = x.shape[: -(inst_rank + 1)] if inst_rank else x.shape[:-1]
    x = x.expand(*prefix_shape, *physical.inst_shape, x.shape[-1])
    with AdcProber(sync_device=torch.device("cpu")) as prober, torch.no_grad():
        physical.vec_mat_mul(x, quantization_mode=quantization_mode, adc_active_bits=None)
    with torch.no_grad():
        ideal_value = ideal.vec_mat_mul(x, quantization_mode=quantization_mode, adc_active_bits=None)

    if not prober.records:
        raise ValueError("the physical macro emitted no ADC input record")
    names = {adc_record.input_name() for adc_record in prober.records}
    if len(names) != 1:
        raise ValueError(f"one probe run emitted multiple ADC input quantities: {sorted(names)}")
    input_parts = [adc_record.input_value().flatten().to("cpu", torch.float32) for adc_record in prober.records]
    input_value = torch.cat(input_parts)
    ideal_value = ideal_value.flatten().to("cpu", torch.int64)
    if input_value.numel() != ideal_value.numel():
        raise ValueError(
            f"paired sample counts differ (ADC input {input_value.numel()} vs ideal {ideal_value.numel()})"
        )
    return names.pop(), input_value, ideal_value


def _collect_random(
    physical: CimMacro[CimMacroConfig, CimMacroPolicy],
    ideal: IdealCimMacro,
    config: AdcProbeToolConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[str, list[Tensor], list[Tensor]]:
    stimulus = config.stimulus
    random = stimulus.random
    weight_values = torch.tensor(_integer_values(physical.w_value_range), dtype=torch.int64, device=device)
    input_values = torch.tensor(_integer_values(physical.x_value_range), dtype=torch.int64, device=device)
    signal_name: str | None = None
    signal_parts: list[Tensor] = []
    ideal_parts: list[Tensor] = []
    weight_batch_num = random.weight_samples // random.batch_w
    for weight_batch in range(weight_batch_num):
        w = sample_values(
            weight_values,
            (random.batch_w, physical.input_num, physical.output_num),
            generator=generator,
        )
        x = sample_sparse_inputs(
            input_values,
            batch_size=random.input_samples_per_weight,
            input_num=physical.input_num,
            max_active_num=physical.max_active_num,
            active_row_selection=stimulus.active_row_selection,
            generator=generator,
        )
        physical.program(w)
        ideal.program(w)
        for input_batch, x_batch in enumerate(x.split(random.batch_x)):
            name, signal, ideal_value = _run_paired(
                physical,
                ideal,
                x_batch.unsqueeze(1),
                quantization_mode=config.probe.quantization_mode,
            )
            if signal_name is None:
                signal_name = name
            elif name != signal_name:
                raise ValueError(f"ADC input name changed across batches: {signal_name!r} -> {name!r}")
            signal_parts.append(signal)
            ideal_parts.append(ideal_value)
            logger.debug(
                "weight batch %d/%d, input batch %d: %d conversion samples",
                weight_batch + 1,
                weight_batch_num,
                input_batch + 1,
                signal.numel(),
            )
    if signal_name is None:
        raise ValueError("ADC probe collected no input batches")
    return signal_name, signal_parts, ideal_parts


def _collect_targeted(
    physical: CimMacro[CimMacroConfig, CimMacroPolicy],
    ideal: IdealCimMacro,
    config: AdcProbeToolConfig,
    target_batch_num: dict[int, int],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[str, list[Tensor], list[Tensor]]:
    stimulus = config.stimulus
    target = stimulus.target
    input_values = _integer_values(physical.x_value_range)
    weight_values = _integer_values(physical.w_value_range)
    target_ideal_values = tuple(target_batch_num)
    sampler = TargetStimulusSampler(
        input_values=input_values,
        weight_values=weight_values,
        active_num=physical.max_active_num,
        input_num=physical.input_num,
        output_num=physical.output_num,
        target_ideal_values=target_ideal_values,
        active_row_selection=stimulus.active_row_selection,
        device=device,
        generator=generator,
    )
    signal_name: str | None = None
    signal_parts: list[Tensor] = []
    ideal_parts: list[Tensor] = []
    for target_value, batch_num in target_batch_num.items():
        for _ in range(batch_num):
            w, x = sampler.sample(target_value, target.batch_w)
            physical.program(w)
            ideal.program(w)
            name, signal, batch_ideal_value = _run_paired(
                physical,
                ideal,
                x,
                quantization_mode=config.probe.quantization_mode,
            )
            if signal_name is None:
                signal_name = name
            elif name != signal_name:
                raise ValueError(f"ADC input name changed across batches: {signal_name!r} -> {name!r}")
            signal_parts.append(signal)
            ideal_parts.append(batch_ideal_value)
        logger.debug(
            "target %d: %d supplemental conversion samples",
            target_value,
            batch_num * target.batch_w * physical.output_num,
        )
    if signal_name is None:
        raise ValueError("ADC probe collected no target batches")
    return signal_name, signal_parts, ideal_parts


def _prepare_probe(
    cfg: AdcProbeToolConfig,
    *,
    run_config_path: Path,
    device: torch.device,
    batch_w: int,
) -> tuple[CimMacro[CimMacroConfig, CimMacroPolicy], IdealCimMacro, tuple[int, ...]]:
    physical = build_physical_macro(
        cfg.macro,
        base=run_config_path,
        device=device,
        inst_shape=(batch_w,),
    )
    _require_all_policy_toggles_off(physical.policy)
    ideal = build_ideal_twin(physical, device=device)
    mode = cfg.probe.quantization_mode
    mode_num = len(physical.config.rescale_factors)
    if not (0 <= mode < mode_num):
        raise ValueError(f"require: quantization_mode ({mode}) in [0, {mode_num})")

    input_values = _integer_values(physical.x_value_range)
    weight_values = _integer_values(physical.w_value_range)
    if 0 not in input_values:
        raise ValueError("require: the macro input value range contains zero so unselected positions are inactive")
    ideal_value_range = _ideal_value_range(
        physical.w_value_range,
        physical.x_value_range,
        physical.max_active_num,
    )
    ideal_value_support = feasible_ideal_values(
        input_values,
        weight_values,
        physical.max_active_num,
    )
    if ideal_value_range != (ideal_value_support[0], ideal_value_support[-1]):
        raise ValueError(
            f"ideal range {ideal_value_range} disagrees with reachable support "
            f"[{ideal_value_support[0]}, {ideal_value_support[-1]}]"
        )
    return physical, ideal, ideal_value_support


def collect_random_probe_data(
    cfg: AdcProbeToolConfig,
    *,
    run_config_path: Path,
    device: torch.device,
) -> AdcProbeData:
    """Collect the unconstrained phase of an ADC-input campaign."""
    physical, ideal, ideal_value_support = _prepare_probe(
        cfg,
        run_config_path=run_config_path,
        device=device,
        batch_w=cfg.stimulus.random.batch_w,
    )
    stimulus = cfg.stimulus
    generator = torch.Generator(device=device).manual_seed(stimulus.random.seed)
    logger.info("uniform weight values: %s", _integer_values(physical.w_value_range))
    logger.info("uniform input values: %s", _integer_values(physical.x_value_range))
    logger.info("max_active_num: %d", physical.max_active_num)
    logger.info("active-row selection: %s", stimulus.active_row_selection.value)
    logger.info("theoretical ideal range: %d .. %d", ideal_value_support[0], ideal_value_support[-1])
    logger.info("reachable signed ideal values: %d", len(ideal_value_support))
    signal_name, signal_parts, ideal_parts = _collect_random(
        physical,
        ideal,
        cfg,
        device=device,
        generator=generator,
    )
    return AdcProbeData(
        input_name=signal_name,
        input_value=torch.cat(signal_parts),
        ideal_value=torch.cat(ideal_parts),
        ideal_value_support=ideal_value_support,
    )


def ideal_value_counts(data: AdcProbeData) -> Tensor:
    """Count every signed ideal value across the theoretical support."""
    lower, upper = data.ideal_value_range
    if data.ideal_value.numel() == 0:
        return torch.zeros(len(data.ideal_value_support), dtype=torch.int64)
    observed = set(torch.unique(data.ideal_value).tolist())
    unexpected = sorted(observed - set(data.ideal_value_support))
    if unexpected:
        raise ValueError(f"observed ideal values are outside the reachable support: {unexpected}")
    range_counts = torch.bincount(data.ideal_value - lower, minlength=upper - lower + 1)
    indices = torch.tensor(data.ideal_value_support, dtype=torch.int64) - lower
    return range_counts[indices]


def plan_target_batches(
    random_data: AdcProbeData,
    *,
    min_samples_per_ideal_value: int,
    samples_per_batch: int,
) -> dict[int, int]:
    """Plan the minimum whole target batches needed to fill random-sample deficits."""
    if min_samples_per_ideal_value <= 0:
        raise ValueError("require: min_samples_per_ideal_value > 0")
    if samples_per_batch <= 0:
        raise ValueError("require: samples_per_batch > 0")
    batches: dict[int, int] = {}
    for ideal_value, count in zip(
        random_data.ideal_value_support,
        ideal_value_counts(random_data).tolist(),
        strict=True,
    ):
        deficit = min_samples_per_ideal_value - count
        if deficit > 0:
            batches[ideal_value] = (deficit + samples_per_batch - 1) // samples_per_batch
    return batches


def collect_targeted_probe_data(
    cfg: AdcProbeToolConfig,
    random_data: AdcProbeData,
    target_batch_num: dict[int, int],
    *,
    run_config_path: Path,
    device: torch.device,
) -> AdcProbeData:
    """Collect only the signed ideal-value deficits left by the random phase."""
    if not target_batch_num:
        return AdcProbeData(
            input_name=random_data.input_name,
            input_value=torch.empty(0, dtype=torch.float32),
            ideal_value=torch.empty(0, dtype=torch.int64),
            ideal_value_support=random_data.ideal_value_support,
        )
    physical, ideal, ideal_value_support = _prepare_probe(
        cfg,
        run_config_path=run_config_path,
        device=device,
        batch_w=cfg.stimulus.target.batch_w,
    )
    if ideal_value_support != random_data.ideal_value_support:
        raise ValueError(
            "reachable ideal values changed between random and target phases: "
            f"{random_data.ideal_value_support} -> {ideal_value_support}"
        )
    generator = torch.Generator(device=device).manual_seed(cfg.stimulus.target.seed)
    signal_name, signal_parts, ideal_parts = _collect_targeted(
        physical,
        ideal,
        cfg,
        target_batch_num,
        device=device,
        generator=generator,
    )
    if signal_name != random_data.input_name:
        raise ValueError(
            f"ADC input name changed between random and target phases: {random_data.input_name!r} -> {signal_name!r}"
        )
    return AdcProbeData(
        input_name=signal_name,
        input_value=torch.cat(signal_parts),
        ideal_value=torch.cat(ideal_parts),
        ideal_value_support=ideal_value_support,
    )
