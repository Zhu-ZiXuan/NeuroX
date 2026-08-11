"""Physical axioms: SI constants (CODATA 2018) and the closed-form laws over them.

See also:
    docs/internals/primitive/physics.md
    docs/reference/primitive/physics.md
"""

from __future__ import annotations

from torch import Tensor

# Boltzmann constant (exact by SI definition).
K_BOLTZMANN__J_per_K: float = 1.380649e-23

# Elementary charge (exact by SI definition).
ELEM_CHARGE__C: float = 1.602176634e-19

# Vacuum permittivity (CODATA 2018).
EPS_0__F_per_m: float = 8.8541878128e-12

# Standard reference temperature (room temperature).
T_ROOM__K: float = 300.0


def thermal_voltage__V(temperature__K: float) -> float:
    """Thermal voltage ``V_T = k_B * T / q``.

    Args:
        temperature__K: Absolute temperature; must be > 0.

    Returns:
        Thermal voltage in SI volts.

    Raises:
        ValueError: If ``temperature__K`` is not positive.
    """
    if temperature__K <= 0.0:
        raise ValueError(f"temperature__K ({temperature__K}) must be > 0")
    return K_BOLTZMANN__J_per_K * temperature__K / ELEM_CHARGE__C


def e_cap_excursion__fJ(v_rail__V: float, c__fF: Tensor | float, delta_v__V: Tensor) -> Tensor:
    """Energy a supply delivers for one excursion of a grounded capacitance.

    Billing is supply-draw, not stored-energy: what a rail spends is the
    charge it hands over times its own potential, ``E = V_rail * C * |delta_v|``,
    which is why the node's own voltage never appears. A node driven from a
    rest level to a working level and back is one round-trip excursion with
    exactly ONE charging leg — the return leg gives its charge to ground and
    draws nothing — so an excursion is billed once, at the magnitude of its
    displacement, whichever direction it travels first.

    ``fF * V^2`` is ``fJ``, so the three arguments already carry the result's
    unit with no conversion factor.

    Args:
        v_rail__V: Potential of the supply that delivers the charge — the
            rail of the driver that owns the node, never the node's own
            level.
        c__fF: Capacitance from the node to ground, broadcastable against
            ``delta_v__V``.
        delta_v__V: Signed displacement of the node between its rest level
            and its working level; only the magnitude is billed.

    Returns:
        Energy drawn from the rail [fJ].
    """
    return v_rail__V * c__fF * delta_v__V.abs()


__all__ = [
    "ELEM_CHARGE__C",
    "EPS_0__F_per_m",
    "K_BOLTZMANN__J_per_K",
    "T_ROOM__K",
    "e_cap_excursion__fJ",
    "thermal_voltage__V",
]
