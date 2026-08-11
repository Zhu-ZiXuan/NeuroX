"""Accumulator billing: one energy quantum per operand element folded in.

:class:`SerialAccumulator` and :class:`Accumulator` share the reduce function
(modular-wrap sum) and the billing law — dynamic energy counts the adder
evaluations, one per operand element, so the reduced extent stays visible in
the energy whichever way the fold is realized. Energy events are captured
under :class:`NeuroxProfiler`.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import EnergyEvent, NeuroxProfiler
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
    return SerialAccumulator(config=_config(bit_width), policy=DigitalPolicy(), inst_shape=inst_shape)


def _build_plain(inst_shape: tuple[int, ...]) -> Accumulator:
    return Accumulator(config=_config(), policy=DigitalPolicy(), inst_shape=inst_shape)


def _energy_total(events: list[EnergyEvent], module: ProfileMixin) -> float:
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted.

    An event payload is a per-unit-operation tensor, so each one totals to its
    own scalar before the events are summed.
    """
    return sum((float(e.dynamic_energy__fJ.sum()) for e in events if e.module is module), 0.0)


def test_serial_accumulator_reduce_matches_plain_accumulator() -> None:
    """Same reduce semantics as ``Accumulator``: exact sum + modular wrap."""
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
    """Energy quanta count equals ``numel(input)``, reduced axis included."""
    torch.manual_seed(12)
    x = torch.randint(-3, 4, (2, 4, 5), dtype=torch.int64)
    serial = _build_serial((2, 5))
    plain = _build_plain((2, 5))
    with NeuroxProfiler() as p:
        serial.accumulate(x, dim=-2)
        plain.accumulate(x, dim=-2)
    expected = _E_OP__FJ * x.numel()
    assert _energy_total(p.energy_events, serial) == pytest.approx(expected)
    assert _energy_total(p.energy_events, plain) == pytest.approx(expected)


def test_accumulate_energy_scales_with_reduced_axis_extent() -> None:
    """At fixed output shape a 4x longer reduced axis costs 4x the accumulate
    energy, on the serial register and on the adder tree alike."""
    out_inst = (2, 5)
    serial = _build_serial(out_inst)
    plain = _build_plain(out_inst)
    energies: dict[int, tuple[float, float]] = {}
    for reduce_extent in (1, 4):
        x = torch.ones((2, reduce_extent, 5), dtype=torch.int64)
        with NeuroxProfiler() as p:
            serial.accumulate(x, dim=-2)
            plain.accumulate(x, dim=-2)
        energies[reduce_extent] = (
            _energy_total(p.energy_events, serial),
            _energy_total(p.energy_events, plain),
        )
    assert energies[1][0] > 0.0
    assert energies[4][0] == pytest.approx(4.0 * energies[1][0])
    assert energies[4][1] == pytest.approx(4.0 * energies[1][1])
