"""Fit the per-mode rescale factor of a CIM macro at `adc_bits`.

Logical weight and input batches are sampled from a configured distribution.
The physical macro and its ideal twin receive identical programs and legal
caller-side active-position planes. The physical result at `adc_bits` and the
twin's highest-precision result already share the logical output layout,
so the fit depends on no ADC implementation or probe. The zero-through-origin
least-squares slope expresses one final full-resolution macro output code in
MAC units.

See Also:
    docs/guides/calibration/calibrate_macro.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neurox.common.module import ConfigBase
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin
from neurox.common.validate_mixin import ValidateMixin
from neurox.primitive.macro.cim import CimMacro, IdealCimMacro
from neurox.tools.config import resolve_relative_path

from ._layout import unroll_active_positions
from .construction import (
    MacroSection,
    build_ideal_twin,
    build_physical_macro,
)
from .math import RescaleFit, fit_rescale_through_origin
from .sampling import load_distribution, make_generator, sample_w, sample_x_batches

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RescaleStimulusConfig(ValidateMixin):
    seed: int
    distribution_file: Path
    weight_samples: int
    input_samples_per_weight: int
    batch_w: int
    batch_x: int

    def __post_init__(self) -> None:
        self._require_pos(self.weight_samples, "[stimulus].weight_samples")
        self._require_pos(self.input_samples_per_weight, "[stimulus].input_samples_per_weight")
        self._require_pos(self.batch_w, "[stimulus].batch_w")
        self._require_pos(self.batch_x, "[stimulus].batch_x")
        if self.weight_samples % self.batch_w:
            raise ValueError("require: weight_samples is divisible by batch_w")
        if self.input_samples_per_weight % self.batch_x:
            raise ValueError("require: input_samples_per_weight is divisible by batch_x")


class RescaleFitToolConfig(ConfigBase):
    macro: MacroSection
    stimulus: RescaleStimulusConfig


class ModeFitResult(TensorDataClassMixin):
    """Full-resolution fit and diagnostics for one operating mode."""

    quantization_mode: int
    adc_bits: int
    fit: RescaleFit
    code: torch.Tensor
    """Macro output code entering the fit.
    Shape: `[sample]`."""
    ideal_value: torch.Tensor
    """Lossless ideal-macro value of the same pairs.
    Shape: `[sample]`."""


def _fit_one_mode(
    physical: CimMacro,
    ideal: IdealCimMacro,
    *,
    quantization_mode: int,
    adc_bits: int,
    stimulus: RescaleStimulusConfig,
    run_config_path: Path,
    device: torch.device,
) -> ModeFitResult:
    """Fit ideal value from one mode's full-resolution macro code."""
    distribution_path = resolve_relative_path(stimulus.distribution_file, run_config_path)
    distribution = load_distribution(distribution_path, physical)
    generator = make_generator(stimulus.seed, torch.device("cpu"))
    code_parts: list[torch.Tensor] = []
    ideal_parts: list[torch.Tensor] = []
    for w in sample_w(
        distribution,
        physical,
        input_num=physical.input_num,
        output_num=physical.output_num,
        n=stimulus.weight_samples,
        batch_w=stimulus.batch_w,
        device=torch.device("cpu"),
        generator=generator,
    ):
        w = w.to(device)
        physical.program(w)
        ideal.program(w)
        for x in sample_x_batches(
            distribution,
            physical,
            input_num=physical.input_num,
            n_total=stimulus.input_samples_per_weight,
            batch_size=stimulus.batch_x,
            device=torch.device("cpu"),
            generator=generator,
        ):
            x = x.unsqueeze(1).to(device)
            planes = unroll_active_positions(
                x,
                input_num=physical.input_num,
                max_active_num=physical.max_active_num,
                inst_shape=physical.inst_shape,
            )
            with torch.no_grad():
                code = physical.vec_mat_mul(
                    planes,
                    quantization_mode=quantization_mode,
                    adc_active_bits=None,
                )
                ideal_value = ideal.vec_mat_mul(
                    planes,
                    quantization_mode=quantization_mode,
                    adc_active_bits=None,
                )
            code_parts.append(code.flatten().to("cpu", torch.float64))
            ideal_parts.append(ideal_value.flatten().to("cpu", torch.float64))
    code = torch.cat(code_parts)
    ideal_value = torch.cat(ideal_parts)
    fit = fit_rescale_through_origin(code, ideal_value)
    return ModeFitResult(
        quantization_mode=quantization_mode,
        adc_bits=adc_bits,
        fit=fit,
        code=code,
        ideal_value=ideal_value,
    )


def _fragment_lines(results: list[ModeFitResult], factors: tuple[float, ...]) -> list[str]:
    """Return one complete rescale-factor assignment."""
    updated = list(factors)
    for result in results:
        updated[result.quantization_mode] = result.fit.rescale_factor
    lines = [
        "# fitted by neurox.tools.calibration.cim_macro.rescale_fit",
        f"# at full ADC resolution: adc_bits = {results[0].adc_bits if results else 0};",
        "# tuple position is quantization_mode",
        "rescale_factors = [",
    ]
    lines.extend(f"    {factor:.6f}," for factor in updated)
    return [*lines, "]"]


@dataclass(frozen=True)
class RescaleResult:
    """Per-mode fits together with the complete original factor assignment."""

    modes: tuple[ModeFitResult, ...]
    original_factors: tuple[float, ...]


def fit_macro_rescale(
    cfg: RescaleFitToolConfig,
    *,
    run_config_path: Path,
    device: torch.device,
    modes: tuple[int, ...] | None = None,
) -> RescaleResult:
    """Fit each selected mode; `None` selects every configured mode.

    Sampling uses the declared workload seed. Physical nonideality randomness
    follows the caller's torch RNG state independently of that workload seed.
    """
    physical = build_physical_macro(
        cfg.macro,
        base=run_config_path,
        device=device,
        inst_shape=(cfg.stimulus.batch_w,),
    )
    ideal = build_ideal_twin(physical, device=device)

    known = tuple(range(len(physical.config.rescale_factors)))
    selected = known if modes is None else modes
    if not selected or len(set(selected)) != len(selected) or any(mode not in known for mode in selected):
        raise ValueError(f"modes must be a nonempty unique subset of {known}")
    # The fit runs at full ADC resolution; every lower active width follows
    # the base-class rescale law from this factor.
    adc_bits = physical.adc_bits
    logger.info(
        "fitting modes %s at full ADC resolution (adc_bits = %d) on %s",
        list(selected),
        adc_bits,
        device,
    )

    results: list[ModeFitResult] = []
    for quantization_mode in selected:
        result = _fit_one_mode(
            physical,
            ideal,
            quantization_mode=quantization_mode,
            adc_bits=adc_bits,
            stimulus=cfg.stimulus,
            run_config_path=run_config_path,
            device=device,
        )
        results.append(result)
        logger.info(
            "mode %d: rescale_factor = %.6f  R^2 = %.6f  rmse = %.4f  max|res| = %.4f  samples = %d",
            quantization_mode,
            result.fit.rescale_factor,
            result.fit.r2,
            result.fit.rmse,
            result.fit.max_abs_residual,
            result.fit.sample_num,
        )

    return RescaleResult(modes=tuple(results), original_factors=physical.config.rescale_factors)


def rescale_fragment_text(result: RescaleResult) -> str:
    """Render a complete factor assignment, preserving unselected modes."""
    return "\n".join(_fragment_lines(list(result.modes), result.original_factors)) + "\n"
