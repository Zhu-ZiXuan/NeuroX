"""Radix folding, modular wrap, and per-digit energy accounting."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.api.profiler import EnergyRecord
from neurox.common.profile_mixin import ProfileMixin
from neurox.primitive.digital import DigitalPolicy, ShiftAdder, ShiftAdderConfig

_E_OP__FJ = 2.5
_T_OP__NS = 3.0


def _config(bit_width: int = 32) -> ShiftAdderConfig:
    return ShiftAdderConfig(
        bit_width=bit_width,
        energy_per_op__fJ=_E_OP__FJ,
        latency_per_op__ns=_T_OP__NS,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _build(
    inst_shape: tuple[int, ...],
    *,
    scale: int,
    digit_count: int,
    bit_width: int = 32,
) -> ShiftAdder:
    unit = ShiftAdder(
        config=_config(bit_width),
        policy=DigitalPolicy(),
        inst_shape=inst_shape,
        scale=scale,
        digit_count=digit_count,
    )
    stamp_names(unit)
    return unit


def _energy_total(records: list[EnergyRecord], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the records `module` emitted."""
    return sum(
        (float(r.dynamic_energy__fJ.sum()) for r in records if r.qualified_name == module.qualified_name),
        0.0,
    )


def test_shift_add_folds_the_digit_axis_by_positional_weight() -> None:
    """Radix-4 digits (1, 2, 3) recombine to 1 + 2*4 + 3*16 = 57."""
    unit = _build((), scale=4, digit_count=3)
    x = torch.tensor([1, 2, 3], dtype=torch.int64)
    assert unit.shift_add(x, dim=0, init_val=None).item() == 57


def test_shift_add_wraps_the_fold_then_adds_the_partial_sum() -> None:
    """A 4-bit register wraps 3 + 3*2 = 9 to -7; `init_val` lands after it."""
    unit = _build((), scale=2, digit_count=2, bit_width=4)
    x = torch.tensor([3, 3], dtype=torch.int64)
    init_val = torch.tensor(10, dtype=torch.int64)
    assert unit.shift_add(x, dim=0, init_val=None).item() == -7
    assert unit.shift_add(x, dim=0, init_val=init_val).item() == 3


def test_shift_add_bills_energy_per_digit_leg() -> None:
    """Energy quanta count equals `numel(input)`, digit axis included."""
    torch.manual_seed(21)
    unit = _build((2, 5), scale=2, digit_count=4)
    x = torch.randint(0, 2, (2, 4, 5), dtype=torch.int64)
    with Profiler() as p:
        unit.shift_add(x, dim=-2, init_val=None)
    assert _energy_total(p.records, unit) == pytest.approx(_E_OP__FJ * x.numel())


def test_shift_add_energy_scales_with_the_digit_count() -> None:
    """At fixed output shape a 4-digit fold costs 4x the single-digit fold."""
    energies: dict[int, float] = {}
    for digit_count in (1, 4):
        unit = _build((2, 5), scale=2, digit_count=digit_count)
        x = torch.ones((2, digit_count, 5), dtype=torch.int64)
        with Profiler() as p:
            unit.shift_add(x, dim=-2, init_val=None)
        energies[digit_count] = _energy_total(p.records, unit)
    assert energies[1] > 0.0
    assert energies[4] == pytest.approx(4.0 * energies[1])


def test_shift_add_partial_sum_preload_is_free() -> None:
    """`init_val` preloads the destination register, so it bills nothing."""
    unit = _build((2, 5), scale=2, digit_count=4)
    x = torch.ones((2, 4, 5), dtype=torch.int64)
    init_val = torch.ones((2, 5), dtype=torch.int64)
    with Profiler() as p:
        unit.shift_add(x, dim=-2, init_val=None)
        unit.shift_add(x, dim=-2, init_val=init_val)
    (bare, preloaded) = (float(e.dynamic_energy__fJ.sum()) for e in p.records)
    assert preloaded == pytest.approx(bare)
