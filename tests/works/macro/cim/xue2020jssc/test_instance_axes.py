"""Eager instance-axis (die) contract for the xue2020jssc SINWP 1T1R CIM macro.

``vec_mat_mul`` declares ``x`` as ``[..., *inst_shape, row_num]``: the instance
axes are part of the leading, not anonymous batch, because every fabricated copy
holds its OWN cells, reference banks and comparator offsets. This file pins that
contract on the hand-built near-ideal witness macro (``_utils.build_config``)
with a ladder calibrated in-code (``_utils.build_calibrated_macro``).

Coverage:

  * per-die weights decode die by die against the CPU int64 oracle, at one and
    two instance axes and with zero, one or two anonymous batch axes ahead of
    them — a fold of the instance block onto any other axis changes the values,
    so bit-exactness per die is the whole law;
  * axes ahead of the instance block stay anonymous batch: reshaping them is
    transparent;
  * a size-1 instance axis shares one input vector across the whole ensemble,
    matching an explicitly expanded input;
  * an ``x`` with no room for the instance axes is rejected;
  * the billed dynamic energy is ADDITIVE over the ensemble: a D-die call bills
    exactly the sum of D independent single-die calls.

Every check is a law (bit-exactness, invariance, additivity), never a shipped
number. Runs eagerly (dynamo disabled) so the ``@torch.compile`` solver leaf is
not unrolled.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler

from ._utils import (
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_K,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    build_calibrated_macro,
    ideal_mac,
)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _per_die_weights(inst_shape: tuple[int, ...]) -> Tensor:
    """Mixed-sign weights, DISTINCT per die, with every output column live.

    Every die rolls the witness matrix along BOTH axes and flips its signs on odd
    dies. Rolling the input axis alone repeats with period 4 and the sign flip
    with period 2, so two dies out of six would share a matrix; the second roll
    advances every two dies, which makes all six distinct. No output column is
    identically zero, so a fold of the instance block onto another axis cannot
    reproduce any die's own MAC on any column.

    Returns:
        Signed weights.
        Shape: ``[*inst_shape, input_num, output_num]``.
    """
    base = torch.tensor(
        [
            [1, 1, -1, 0],
            [-2, 1, 0, 0],
            [3, -1, 0, 0],
            [0, 1, -1, 0],
        ],
        dtype=torch.long,
    ).transpose(-1, -2)
    dies = math.prod(inst_shape)
    stacked = torch.stack(
        [base.roll(shifts=d, dims=0).roll(shifts=d // 2, dims=1) * (1 - 2 * (d % 2)) for d in range(dies)]
    )
    return stacked.reshape(*inst_shape, TINY_INPUT_NUM, TINY_OUTPUT_NUM)


def _die_inputs(batch: tuple[int, ...], inst_shape: tuple[int, ...]) -> Tensor:
    """Distinct K-bit activation vectors, one per (batch, die) slot.

    Returns:
        Integer activations.
        Shape: ``[*batch, *inst_shape, input_num]``.
    """
    total = math.prod(batch) * math.prod(inst_shape) * TINY_INPUT_NUM
    values = (torch.arange(total, dtype=torch.long) * 3) % (1 << TINY_K)
    return values.reshape(*batch, *inst_shape, TINY_INPUT_NUM)


def _programmed(device: torch.device, inst_shape: tuple[int, ...]) -> tuple[Xue2020JsscCimMacro, Tensor]:
    """Calibrated macro at ``inst_shape``, programmed with the per-die weights."""
    macro = build_calibrated_macro(device=device, inst_shape=inst_shape)
    w = _per_die_weights(inst_shape)
    macro.program(w.to(device))
    return macro, w


def _run(macro: Xue2020JsscCimMacro, x: Tensor) -> Tensor:
    """One VMM at the witness resolution; codes on CPU."""
    with torch.no_grad():
        out = macro.vec_mat_mul(
            x.to(next(macro.buffers()).device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS
        )
    return out.cpu()


@pytest.mark.parametrize("inst_shape", [(2,), (2, 3)])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_per_die_weights_decode_die_by_die(
    device: torch.device, inst_shape: tuple[int, ...], batch: tuple[int, ...]
) -> None:
    """Each die decodes its OWN weights against its OWN input vector, bit-exactly."""
    macro, w = _programmed(device, inst_shape)
    x = _die_inputs(batch, inst_shape)

    out = _run(macro, x)
    assert tuple(out.shape) == (*batch, *inst_shape, TINY_OUTPUT_NUM)

    for die in itertools.product(*(range(n) for n in inst_shape)):
        key = (slice(None),) * len(batch) + die
        expected = ideal_mac(w[die], x[key])
        assert torch.equal(out[key], expected), f"die {die} mismatch:\n{out[key].tolist()}\nvs\n{expected.tolist()}"
    # The dies must not agree by accident — distinct weights, distinct codes.
    assert not torch.equal(out[(slice(None),) * len(batch) + (0,)], out[(slice(None),) * len(batch) + (1,)])


def test_batch_reshape_transparent_ahead_of_instance_axes(device: torch.device) -> None:
    """Axes AHEAD of the instance block stay anonymous batch: reshaping them is transparent."""
    inst_shape = (2,)
    macro, _w = _programmed(device, inst_shape)
    x_flat = _die_inputs((6,), inst_shape)

    out_flat = _run(macro, x_flat)
    out_multi = _run(macro, x_flat.reshape(2, 3, *inst_shape, TINY_INPUT_NUM))
    assert tuple(out_multi.shape) == (2, 3, *inst_shape, TINY_OUTPUT_NUM)
    assert torch.equal(out_multi, out_flat.reshape(2, 3, *inst_shape, TINY_OUTPUT_NUM))

    # A bare instance-only input yields the instance block plus the output axis.
    out_bare = _run(macro, x_flat[0])
    assert tuple(out_bare.shape) == (*inst_shape, TINY_OUTPUT_NUM)
    assert torch.equal(out_bare, out_flat[0])


def test_size_one_instance_axis_shares_one_input_vector(device: torch.device) -> None:
    """A size-1 instance axis broadcasts one input vector across the whole ensemble."""
    inst_shape = (3,)
    macro, w = _programmed(device, inst_shape)
    # Shape: [2, 1, input_num] — one vector per batch slot, shared by every die.
    x_shared = _die_inputs((2,), (1,))

    out = _run(macro, x_shared)
    assert tuple(out.shape) == (2, *inst_shape, TINY_OUTPUT_NUM)
    for die in range(inst_shape[0]):
        assert torch.equal(out[:, die], ideal_mac(w[die], x_shared[:, 0]))
    # Identical to materializing the same vector on every die.
    assert torch.equal(out, _run(macro, x_shared.expand(2, *inst_shape, TINY_INPUT_NUM).contiguous()))


def test_missing_instance_axis_rejected(device: torch.device) -> None:
    """An ``x`` with no room for the instance axes is rejected, not silently folded."""
    macro, _w = _programmed(device, (2,))
    x = torch.zeros(TINY_INPUT_NUM, dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="inst_shape"), torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)


def test_dynamic_energy_is_additive_over_the_ensemble(device: torch.device) -> None:
    """A D-die call bills exactly the sum of D independent single-die calls."""
    inst_shape = (3,)
    macro, w = _programmed(device, inst_shape)
    x = _die_inputs((), inst_shape)

    with NeuroxProfiler() as prof, torch.no_grad():
        macro.vec_mat_mul(x.to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    ensemble__fJ = prof.total_dynamic_energy__fJ

    separate__fJ = 0.0
    for die in range(inst_shape[0]):
        one = build_calibrated_macro(device=device)
        one.program(w[die].to(device))
        with NeuroxProfiler() as prof_one, torch.no_grad():
            one.vec_mat_mul(x[die].to(device), quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
        separate__fJ += prof_one.total_dynamic_energy__fJ

    assert ensemble__fJ == pytest.approx(separate__fJ)
    assert separate__fJ > 0.0
