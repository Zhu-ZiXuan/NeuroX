"""SI constants and closed-form charge and supply-energy laws.

See Also:
    docs/reference/primitive/physics.md
"""

from __future__ import annotations

from typing import overload

from torch import Tensor

__all__ = [
    "ELEM_CHARGE__fC",
    "EPS_0__fF_per_um",
    "K_BOLTZMANN__fJ_per_K",
    "delta_q_cap__fC",
    "e_cap__fJ",
    "e_cap_excursion__fJ",
    "e_charge__fJ",
    "q_conduction__fC",
    "thermal_fluctuation_energy__fJ",
    "thermal_voltage__V",
]


# ### Physical constants ###


# Boltzmann constant (exact by SI definition).
K_BOLTZMANN__fJ_per_K: float = 1.380649e-8

# Elementary charge (exact by SI definition).
ELEM_CHARGE__fC: float = 1.602176634e-4

# Vacuum permittivity (CODATA 2018).
EPS_0__fF_per_um: float = 8.8541878128e-3


# ### Thermal ###


def thermal_fluctuation_energy__fJ(T__K: float) -> float:
    """Thermal energy `E_T = k_B · T`."""
    return K_BOLTZMANN__fJ_per_K * T__K


def thermal_voltage__V(T__K: float) -> float:
    """Thermal voltage `V_T = k_B · T / q`."""
    return thermal_fluctuation_energy__fJ(T__K) / ELEM_CHARGE__fC


# ### Charge ###


@overload
def q_conduction__fC(
    i__uA: float,
    duration__ns: float,
) -> float: ...


@overload
def q_conduction__fC(
    i__uA: Tensor,
    duration__ns: float,
) -> Tensor: ...


def q_conduction__fC(
    i__uA: Tensor | float,
    duration__ns: float,
) -> Tensor | float:
    """Charge transferred by a constant branch current, `q = I · t`."""
    return i__uA * duration__ns


@overload
def delta_q_cap__fC(
    c__fF: float,
    delta_v__V: float,
) -> float: ...


@overload
def delta_q_cap__fC(
    c__fF: float,
    delta_v__V: Tensor,
) -> Tensor: ...


@overload
def delta_q_cap__fC(
    c__fF: Tensor,
    delta_v__V: Tensor | float,
) -> Tensor: ...


def delta_q_cap__fC(
    c__fF: Tensor | float,
    delta_v__V: Tensor | float,
) -> Tensor | float:
    """Signed charge change of a constant capacitance.

    Inputs follow tensor broadcasting rules. The result retains the sign of
    the voltage change for nonnegative capacitance.
    """
    return c__fF * delta_v__V


# ### Supply energy ###


@overload
def e_charge__fJ(
    v_supply__V: float,
    delta_q_abs__fC: float,
) -> float: ...


@overload
def e_charge__fJ(
    v_supply__V: float,
    delta_q_abs__fC: Tensor,
) -> Tensor: ...


def e_charge__fJ(
    v_supply__V: float,
    delta_q_abs__fC: Tensor | float,
) -> Tensor | float:
    """Supply energy consumed by transporting a charge magnitude.

    The caller supplies the absolute charge change; no absolute value,
    clamping, or input validation is applied.
    """
    return v_supply__V * delta_q_abs__fC


@overload
def e_cap__fJ(
    v_supply__V: float,
    c__fF: float,
    delta_v_abs__V: float,
) -> float: ...


@overload
def e_cap__fJ(
    v_supply__V: float,
    c__fF: Tensor,
    delta_v_abs__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap__fJ(
    v_supply__V: float,
    c__fF: float,
    delta_v_abs__V: Tensor,
) -> Tensor: ...


def e_cap__fJ(
    v_supply__V: float,
    c__fF: Tensor | float,
    delta_v_abs__V: Tensor | float,
) -> Tensor | float:
    """Supply energy consumed for a grounded capacitance's voltage change.

    The caller supplies nonnegative capacitance and the absolute voltage
    change, and owns the billing of each excursion. No absolute value,
    clamping, or input validation is applied. Inputs follow tensor
    broadcasting rules.

    Args:
        v_supply__V: Potential of the supply that delivers the charge — the
            rail of the driver that owns the node, never the node's own
            level.
    """
    return v_supply__V * c__fF * delta_v_abs__V


@overload
def e_cap_excursion__fJ(
    v_supply__V: float,
    c__fF: float,
    *,
    v_rest__V: float,
    v_work__V: float,
) -> float: ...


@overload
def e_cap_excursion__fJ(
    v_supply__V: float,
    c__fF: Tensor,
    *,
    v_rest__V: Tensor | float,
    v_work__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap_excursion__fJ(
    v_supply__V: float,
    c__fF: float,
    *,
    v_rest__V: Tensor,
    v_work__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap_excursion__fJ(
    v_supply__V: float,
    c__fF: float,
    *,
    v_rest__V: float,
    v_work__V: Tensor,
) -> Tensor: ...


def e_cap_excursion__fJ(
    v_supply__V: float,
    c__fF: Tensor | float,
    *,
    v_rest__V: Tensor | float,
    v_work__V: Tensor | float,
) -> Tensor | float:
    """Supply energy for one rest-to-work-to-rest capacitive excursion.

    One call accounts for the complete excursion in either voltage direction.
    The caller supplies nonnegative capacitance and the supply potential behind
    that node. Inputs broadcast; the result preserves their broadcast shape.
    """
    delta_v__V = v_work__V - v_rest__V
    # Scalar versus tensor is fixed during tracing.
    delta_v_abs__V = delta_v__V.abs() if isinstance(delta_v__V, Tensor) else abs(delta_v__V)
    return e_cap__fJ(v_supply__V, c__fF, delta_v_abs__V)
