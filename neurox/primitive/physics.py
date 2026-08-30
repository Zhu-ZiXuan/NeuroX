"""Physical axioms: SI constants (CODATA 2018) and the closed-form laws over them.

A device- or circuit-level physical parameter comes from configuration; these constants are
the sole exception, fixed because they vary with neither device, process, nor chip. A law
that takes plain numbers, owns no configuration, and holds for every scheme is an axiom of
the same kind, so it lives here beside them rather than in a module of its own.

See Also:
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
    """Thermal voltage `V_T = k_B · T / q`.

    Raises:
        ValueError: Temperature is not positive.
    """
    if temperature__K <= 0.0:
        raise ValueError(f"temperature__K ({temperature__K}) must be > 0")
    return K_BOLTZMANN__J_per_K * temperature__K / ELEM_CHARGE__C


def q_conduction__fC(i__uA: Tensor, duration__ns: float) -> Tensor:
    """Charge transferred by a constant branch current, `q = I · t`."""
    return i__uA * duration__ns


def e_supply_charge__fJ(v_rail__V: float, q__fC: Tensor) -> Tensor:
    """Energy delivered by a supply rail with charge `q`, `E = V · q`."""
    return v_rail__V * q__fC


def e_cap_excursion__fJ(v_rail__V: float, c__fF: Tensor | float, delta_v__V: Tensor) -> Tensor:
    """Energy a supply delivers for one excursion of a grounded capacitance.

    Supply-draw billing `E = V_rail · C · |Δv|`, charged once per round-trip
    excursion regardless of which leg travels first.

    Args:
        v_rail__V: Potential of the supply that delivers the charge — the
            rail of the driver that owns the node, never the node's own
            level.
        c__fF: Capacitance from the node to ground, broadcastable against
            `delta_v__V`.
        delta_v__V: Signed displacement of the node between its rest level
            and its working level; only the magnitude is billed.
    """
    return v_rail__V * c__fF * delta_v__V.abs()


__all__ = [
    "ELEM_CHARGE__C",
    "EPS_0__F_per_m",
    "K_BOLTZMANN__J_per_K",
    "T_ROOM__K",
    "e_cap_excursion__fJ",
    "e_supply_charge__fJ",
    "q_conduction__fC",
    "thermal_voltage__V",
]
