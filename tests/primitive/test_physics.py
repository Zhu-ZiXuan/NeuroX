"""Laws of the supply-draw capacitive energy primitive.

`e_cap_excursion__fJ` states one rule — a rail delivers `C * |delta_v|`
of charge at its own potential, once per excursion — so the tests pin the
consequences of that rule rather than any number: the bill is blind to the
direction of travel and to the node's absolute level, it is linear in each of
the three factors independently, an excursion of zero costs nothing, and a
per-position capacitance broadcasts against a displacement grid.

Tiny CPU shapes; no module, config, or policy is involved.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.physics import e_cap_excursion__fJ, e_supply_charge__fJ, q_conduction__fC

_DTYPE = torch.float64
_V_RAIL__V = 0.9
_C__fF = 1.7


def test_conduction_charge_and_supply_energy_use_the_runtime_unit_scales() -> None:
    i__uA = torch.tensor([0.5, 2.0], dtype=_DTYPE)
    duration__ns = 3.0

    q__fC = q_conduction__fC(i__uA, duration__ns)
    e__fJ = e_supply_charge__fJ(_V_RAIL__V, q__fC)

    torch.testing.assert_close(q__fC, i__uA * duration__ns)
    torch.testing.assert_close(e__fJ, _V_RAIL__V * i__uA * duration__ns)


def test_bill_is_the_rail_times_the_charge_moved() -> None:
    """`E == V_rail * C * |delta_v|` elementwise, on an explicit witness."""
    delta_v__V = torch.tensor([0.25, -0.5, 0.0], dtype=_DTYPE)

    got = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, delta_v__V)

    expected = torch.tensor(
        [_V_RAIL__V * _C__fF * 0.25, _V_RAIL__V * _C__fF * 0.5, 0.0],
        dtype=_DTYPE,
    )
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_one_charging_leg_per_excursion_is_direction_blind() -> None:
    """Charging up and discharging down cost the same: one leg is billed, not two."""
    delta_v__V = torch.tensor([0.3, 0.05, 1.2], dtype=_DTYPE)

    up = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, delta_v__V)
    down = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, -delta_v__V)

    assert torch.equal(up, down)


def test_the_node_level_never_enters_the_bill() -> None:
    """Two excursions of equal length cost the same wherever they sit."""
    low = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, torch.tensor([0.4], dtype=_DTYPE) - 0.0)
    high = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, torch.tensor([0.9], dtype=_DTYPE) - 0.5)

    assert torch.equal(low, high)


def test_a_still_node_is_free() -> None:
    """No displacement, no charge handed over, no bill."""
    still = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, torch.zeros(4, dtype=_DTYPE))

    assert torch.equal(still, torch.zeros(4, dtype=_DTYPE))


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_bill_is_linear_in_each_factor_independently(factor: float) -> None:
    """Scaling the rail, the capacitance, or the displacement scales the bill alike."""
    delta_v__V = torch.tensor([0.3, -0.7], dtype=_DTYPE)
    base = e_cap_excursion__fJ(_V_RAIL__V, _C__fF, delta_v__V)

    torch.testing.assert_close(e_cap_excursion__fJ(factor * _V_RAIL__V, _C__fF, delta_v__V), factor * base)
    torch.testing.assert_close(e_cap_excursion__fJ(_V_RAIL__V, factor * _C__fF, delta_v__V), factor * base)
    torch.testing.assert_close(e_cap_excursion__fJ(_V_RAIL__V, _C__fF, factor * delta_v__V), abs(factor) * base)


def test_a_per_position_capacitance_broadcasts_against_the_grid() -> None:
    """A capacitance vector bills its own axis of a displacement grid."""
    # Shape: [segment_num]
    c__fF = torch.tensor([1.0, 2.0, 4.0], dtype=_DTYPE)
    # Shape: [position_num, segment_num]
    delta_v__V = torch.tensor([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=_DTYPE)

    got = e_cap_excursion__fJ(_V_RAIL__V, c__fF, delta_v__V)

    assert got.shape == delta_v__V.shape
    torch.testing.assert_close(got, _V_RAIL__V * c__fF * delta_v__V.abs())
