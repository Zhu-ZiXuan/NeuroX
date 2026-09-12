"""Charge conversion, supply-energy scaling, and capacitive broadcasting laws."""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.physics import delta_q_cap__fC, e_cap__fJ, e_cap_excursion__fJ, e_charge__fJ, q_conduction__fC

_DTYPE = torch.float64
_V_RAIL__V = 0.9
_C__fF = 1.7


def test_conduction_charge_and_supply_energy_use_the_runtime_unit_scales() -> None:
    i__uA = torch.tensor([0.5, 2.0], dtype=_DTYPE)
    duration__ns = 3.0

    q__fC = q_conduction__fC(i__uA, duration__ns)
    e__fJ = e_charge__fJ(_V_RAIL__V, q__fC)

    torch.testing.assert_close(q__fC, i__uA * duration__ns)
    torch.testing.assert_close(e__fJ, _V_RAIL__V * i__uA * duration__ns)


def test_bill_is_the_rail_times_the_charge_moved() -> None:
    """`E == V_rail * C * |delta_v|` elementwise, on an explicit witness."""
    delta_v__V = torch.tensor([0.25, -0.5, 0.0], dtype=_DTYPE)

    got = e_cap__fJ(_V_RAIL__V, _C__fF, delta_v__V.abs())

    expected = torch.tensor(
        [_V_RAIL__V * _C__fF * 0.25, _V_RAIL__V * _C__fF * 0.5, 0.0],
        dtype=_DTYPE,
    )
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_plain_numbers_return_plain_energy() -> None:
    """A scalar circuit constant stays a Python number until tensor arithmetic needs it."""
    delta_v__V = -0.25

    got = e_cap__fJ(_V_RAIL__V, _C__fF, abs(delta_v__V))

    assert isinstance(got, float)
    assert got == pytest.approx(_V_RAIL__V * _C__fF * abs(delta_v__V))


def test_capacitive_energy_matches_charge_magnitude_energy() -> None:
    delta_v__V = torch.tensor([0.3, -0.05, 0.0], dtype=_DTYPE)
    delta_q__fC = delta_q_cap__fC(_C__fF, delta_v__V)

    from_voltage = e_cap__fJ(_V_RAIL__V, _C__fF, delta_v__V.abs())
    from_charge = e_charge__fJ(_V_RAIL__V, delta_q__fC.abs())

    torch.testing.assert_close(from_voltage, from_charge)


def test_the_node_level_never_enters_the_bill() -> None:
    """Two excursions of equal length cost the same wherever they sit."""
    low = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, v_rest__V=0.0, v_work__V=torch.tensor([0.4], dtype=_DTYPE))
    high = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, v_rest__V=0.5, v_work__V=torch.tensor([0.9], dtype=_DTYPE))

    assert torch.equal(low, high)


def test_a_still_node_is_free() -> None:
    """No displacement, no charge handed over, no bill."""
    levels = torch.tensor([-0.5, 0.0, 0.25, 0.75], dtype=_DTYPE)
    still = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, v_rest__V=levels, v_work__V=levels)

    assert torch.equal(still, torch.zeros(4, dtype=_DTYPE))


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_bill_is_linear_in_each_factor_independently(factor: float) -> None:
    """Scaling the rail, the capacitance, or the displacement scales the bill alike."""
    delta_v_abs__V = torch.tensor([0.3, 0.7], dtype=_DTYPE)
    base = e_cap__fJ(_V_RAIL__V, _C__fF, delta_v_abs__V)

    torch.testing.assert_close(e_cap__fJ(factor * _V_RAIL__V, _C__fF, delta_v_abs__V), factor * base)
    torch.testing.assert_close(e_cap__fJ(_V_RAIL__V, factor * _C__fF, delta_v_abs__V), factor * base)
    torch.testing.assert_close(e_cap__fJ(_V_RAIL__V, _C__fF, factor * delta_v_abs__V), factor * base)


def test_a_per_position_capacitance_broadcasts_against_the_grid() -> None:
    """A capacitance vector bills its own axis of a displacement grid."""
    # Shape: [segment]
    c__fF = torch.tensor([1.0, 2.0, 4.0], dtype=_DTYPE)
    # Shape: [position, segment]
    delta_v__V = torch.tensor([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=_DTYPE)

    got = e_cap__fJ(_V_RAIL__V, c__fF, delta_v__V.abs())

    assert got.shape == delta_v__V.shape
    torch.testing.assert_close(got, _V_RAIL__V * c__fF * delta_v__V.abs())


@pytest.mark.parametrize(("v_rest__V", "v_work__V"), [(0.25, 0.75), (0.75, 0.25), (0.25, -0.25)])
def test_scalar_excursion_bills_one_charging_leg(v_rest__V: float, v_work__V: float) -> None:
    energy__fJ = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, v_rest__V=v_rest__V, v_work__V=v_work__V)

    assert isinstance(energy__fJ, float)
    assert energy__fJ == pytest.approx(_V_RAIL__V * _C__fF * 0.5)


@pytest.mark.parametrize("compiled", [False, True])
def test_excursion_broadcasting_preserves_direction_symmetry(compiled: bool) -> None:
    c__fF = torch.tensor([1.0, 2.0, 4.0], dtype=_DTYPE)
    rest = torch.tensor([[0.25], [0.75]], dtype=_DTYPE)
    work = torch.tensor([0.0, 0.5, 1.0], dtype=_DTYPE)

    def evaluate(v_rest: torch.Tensor, v_work: torch.Tensor) -> torch.Tensor:
        return e_cap_excursion__fJ(_V_RAIL__V, c__fF, v_rest__V=v_rest, v_work__V=v_work)

    run = torch.compile(evaluate, fullgraph=True) if compiled else evaluate
    energy = run(rest, work)
    expected = e_charge__fJ(_V_RAIL__V, delta_q_cap__fC(c__fF, work - rest).abs())

    assert energy.shape == (2, 3)
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(run(work, rest), energy)
