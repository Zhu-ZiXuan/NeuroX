"""Xue2020 leading-axis broadcast tests."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch import Tensor

from neurox import Profiler
from tests.works.macro.cim.xue2020jssc.macro._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_K,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    build_calibrated_macro,
    ideal_mac,
)


def _per_die_weights(inst_shape: tuple[int, ...]) -> Tensor:
    """Return distinct mixed-sign weights for every instance."""
    base = torch.tensor(
        [
            [1, 1, -1, 0],
            [-2, 1, 0, 0],
            [3, -1, 0, 0],
            [0, 1, -1, 0],
        ],
        dtype=torch.int32,
    ).transpose(-1, -2)
    dies = math.prod(inst_shape)
    stacked = torch.stack(
        [base.roll(shifts=d, dims=0).roll(shifts=d // 2, dims=1) * (1 - 2 * (d % 2)) for d in range(dies)]
    )
    return stacked.reshape(*inst_shape, TINY_INPUT_NUM, TINY_OUTPUT_NUM)


def _die_inputs(batch: tuple[int, ...], inst_shape: tuple[int, ...]) -> Tensor:
    """Return distinct activations for every batch-instance slot."""
    total = math.prod(batch) * math.prod(inst_shape) * TINY_INPUT_NUM
    values = (torch.arange(total, dtype=torch.int32) * 3) % (1 << TINY_K)
    return values.reshape(*batch, *inst_shape, TINY_INPUT_NUM)


def _programmed(device: torch.device, inst_shape: tuple[int, ...]) -> tuple[Xue2020JsscCimMacro, Tensor]:
    """Calibrated macro at `inst_shape`, programmed with the per-die weights."""
    macro = build_calibrated_macro(device=device, inst_shape=inst_shape)
    w = _per_die_weights(inst_shape)
    macro.program(w.to(device))
    return macro, w


def _run(macro: Xue2020JsscCimMacro, x: Tensor) -> Tensor:
    """One VMM at the witness resolution; codes on CPU."""
    with torch.no_grad():
        out = macro.vec_mat_mul(
            x.to(next(macro.buffers()).device),
            quantization_mode=QUANTIZATION_MODE,
            adc_active_bits=TINY_ADC_BITS,
        )
    return out.cpu()


@pytest.mark.parametrize("inst_shape", [(2,), (2, 3)])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_per_die_weights_decode_die_by_die(
    device: torch.device, inst_shape: tuple[int, ...], batch: tuple[int, ...]
) -> None:
    macro, w = _programmed(device, inst_shape)
    x = _die_inputs(batch, inst_shape)

    out = _run(macro, x)
    assert tuple(out.shape) == (*batch, *inst_shape, TINY_OUTPUT_NUM)

    for die in itertools.product(*(range(n) for n in inst_shape)):
        key = (slice(None),) * len(batch) + die
        expected = ideal_mac(w[die], x[key])
        assert torch.equal(out[key], expected), f"die {die} mismatch:\n{out[key].tolist()}\nvs\n{expected.tolist()}"
    # Keep the per-instance comparison non-vacuous.
    assert not torch.equal(out[(slice(None),) * len(batch) + (0,)], out[(slice(None),) * len(batch) + (1,)])


def test_dynamic_energy_is_additive_over_the_ensemble(device: torch.device) -> None:
    inst_shape = (3,)
    macro, w = _programmed(device, inst_shape)
    x = _die_inputs((), inst_shape)

    with Profiler(concat_dim=0) as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_active_bits=TINY_ADC_BITS)
    ensemble__fJ = sum(
        item.dynamic_energy__fJ.sum().item() for item in prof.result.values() if item.dynamic_energy__fJ is not None
    )

    separate__fJ = 0.0
    for die in range(inst_shape[0]):
        one = build_calibrated_macro(device=device)
        one.program(w[die].to(device))
        with Profiler(concat_dim=0) as prof_one, torch.no_grad():
            one.vec_mat_mul(
                x[die].to(device),
                quantization_mode=QUANTIZATION_MODE,
                adc_active_bits=TINY_ADC_BITS,
            )
        separate__fJ += sum(
            item.dynamic_energy__fJ.sum().item()
            for item in prof_one.result.values()
            if item.dynamic_energy__fJ is not None
        )

    assert ensemble__fJ == pytest.approx(separate__fJ)
    assert separate__fJ > 0.0
