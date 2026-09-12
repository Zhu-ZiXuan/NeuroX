"""Accumulator modular sums and energy scaling with operand count."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox import Profiler, stamp_names
from neurox.api.profiler import EnergyRecord
from neurox.common.profile_mixin import ProfileMixin
from neurox.primitive.digital import (
    Accumulator,
    AccumulatorConfig,
    DigitalPolicy,
    SerialAccumulator,
)

_E_OP__FJ = 2.5
_T_OP__NS = 3.0


def _config(bit_width: int = 32) -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=bit_width,
        energy_per_op__fJ=_E_OP__FJ,
        latency_per_op__ns=_T_OP__NS,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    )


def _build_serial(inst_shape: tuple[int, ...], *, bit_width: int = 32) -> SerialAccumulator:
    serial = SerialAccumulator(config=_config(bit_width), policy=DigitalPolicy(), inst_shape=inst_shape)
    stamp_names(serial)
    return serial


def _build_plain(inst_shape: tuple[int, ...]) -> Accumulator:
    plain = Accumulator(config=_config(), policy=DigitalPolicy(), inst_shape=inst_shape)
    stamp_names(plain)
    return plain


def _build_pair(inst_shape: tuple[int, ...]) -> tuple[SerialAccumulator, Accumulator]:
    """Bind both accumulators in one tree, so one walk names their rows apart."""
    serial = _build_serial(inst_shape)
    plain = _build_plain(inst_shape)
    pair = nn.Module()
    pair.serial = serial
    pair.plain = plain
    stamp_names(pair)
    return serial, plain


def _energy_total(records: list[EnergyRecord], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the records `module` emitted.

    A record's energy is a per-unit-operation tensor, so each one totals to its
    own scalar before the records are summed. A record carries the name its tree
    stamped, so telling `serial` and `plain` apart is a matter of binding
    them in one tree that names them both.
    """
    return sum(
        (float(r.dynamic_energy__fJ.sum()) for r in records if r.qualified_name == module.qualified_name),
        0.0,
    )


def test_serial_accumulator_reduce_matches_plain_accumulator() -> None:
    """Same reduce semantics as `Accumulator`: exact sum + modular wrap."""
    torch.manual_seed(11)
    x = torch.randint(-100, 100, (3, 4, 5), dtype=torch.int64)
    serial = _build_serial((3, 5))
    plain = _build_plain((3, 5))
    assert torch.equal(serial.accumulate(x, dim=-2), plain.accumulate(x, dim=-2))


def test_serial_accumulator_wraps_modulo_bit_width() -> None:
    """A 4-bit register wraps 7 + 7 = 14 to -2 in two's-complement."""
    acc = _build_serial((), bit_width=4)
    x = torch.tensor([7, 7], dtype=torch.int64)
    assert acc.accumulate(x, dim=0).item() == -2


def test_accumulate_bills_energy_per_operand_element() -> None:
    """Energy quanta count equals `numel(input)`, reduced axis included."""
    torch.manual_seed(12)
    x = torch.randint(-3, 4, (2, 4, 5), dtype=torch.int64)
    serial, plain = _build_pair((2, 5))
    with Profiler() as p:
        serial.accumulate(x, dim=-2)
        plain.accumulate(x, dim=-2)
    expected = _E_OP__FJ * x.numel()
    assert _energy_total(p.records, serial) == pytest.approx(expected)
    assert _energy_total(p.records, plain) == pytest.approx(expected)


def test_accumulate_energy_scales_with_reduced_axis_extent() -> None:
    """At fixed output shape a 4x longer reduced axis costs 4x the accumulate
    energy, on the serial register and on the adder tree alike."""
    out_inst = (2, 5)
    serial, plain = _build_pair(out_inst)
    energies: dict[int, tuple[float, float]] = {}
    for reduce_extent in (1, 4):
        x = torch.ones((2, reduce_extent, 5), dtype=torch.int64)
        with Profiler() as p:
            serial.accumulate(x, dim=-2)
            plain.accumulate(x, dim=-2)
        energies[reduce_extent] = (
            _energy_total(p.records, serial),
            _energy_total(p.records, plain),
        )
    assert energies[1][0] > 0.0
    assert energies[4][0] == pytest.approx(4.0 * energies[1][0])
    assert energies[4][1] == pytest.approx(4.0 * energies[1][1])
