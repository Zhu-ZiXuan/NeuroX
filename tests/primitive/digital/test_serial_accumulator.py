"""SerialAccumulator billing: per-input energy quanta, per-input serial latency.

The reduce function (modular-wrap sum) is identical to :class:`Accumulator`;
only the billing law differs — the reduced axis is a time-serial operand
stream, so dynamic energy carries one quantum per INPUT element and the
serial-op count divides the input-element count across instances, where the
plain accumulator bills both against the output-element count.
Events are captured under :class:`NeuroxProfiler`.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import EnergyEvent, LatencyEvent, NeuroxProfiler
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
    """Sum the logged dynamic energy [fJ] of the events ``module`` emitted."""
    return sum(e.dynamic_energy__fJ for e in events if e.module is module)


def _latency_total(events: list[LatencyEvent], module: ProfileMixin) -> float:
    """Sum the logged latency [ns] of the events ``module`` emitted."""
    return sum(e.latency__ns for e in events if e.module is module)


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


def test_serial_accumulator_bills_energy_per_input_element() -> None:
    """Energy quanta count equals ``numel(input)``, reduced axis included."""
    torch.manual_seed(12)
    x = torch.randint(-3, 4, (2, 4, 5), dtype=torch.int64)
    acc = _build_serial((2, 5))
    with NeuroxProfiler() as p:
        acc.accumulate(x, dim=-2)
    assert _energy_total(p.energy_events, acc) == pytest.approx(_E_OP__FJ * x.numel())


def test_serial_accumulator_energy_scales_with_reduced_axis_extent() -> None:
    """At fixed output shape a 4x longer reduced axis costs 4x the accumulate
    energy; the plain ``Accumulator``'s per-output billing is invariant."""
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
    assert energies[4][1] == pytest.approx(energies[1][1])


def test_serial_accumulator_latency_counts_serial_inputs_per_instance() -> None:
    """Latency = ``latency_per_op * ceil(numel(input) / inst_count)``."""
    x = torch.ones((2, 4, 5), dtype=torch.int64)  # 40 inputs
    acc = _build_serial((2, 5))  # inst_count 10 -> 4 serial ops
    with NeuroxProfiler() as p:
        acc.accumulate(x, dim=-2)
    assert _latency_total(p.latency_events, acc) == pytest.approx(_T_OP__NS * 4)


def test_serial_accumulator_latency_ceils_partial_instance_load() -> None:
    """A non-divisible input count rounds the serial-op count up."""
    x = torch.ones((3, 2, 5), dtype=torch.int64)  # 30 inputs
    acc = _build_serial((4,))  # inst_count 4 -> ceil(30 / 4) = 8
    with NeuroxProfiler() as p:
        acc.accumulate(x, dim=-2)
    assert _latency_total(p.latency_events, acc) == pytest.approx(_T_OP__NS * 8)
