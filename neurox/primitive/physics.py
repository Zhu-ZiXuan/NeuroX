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
    """Return thermal fluctuation energy at a positive absolute temperature.

    This scalar helper applies the configured-unit Boltzmann constant without
    validating temperature. Callers must supply a finite, physically valid
    value.

    Args:
        T__K: Finite positive absolute temperature.

    Returns:
        Thermal fluctuation energy as a Python float.
    """
    return K_BOLTZMANN__fJ_per_K * T__K


def thermal_voltage__V(T__K: float) -> float:
    """Return thermal voltage at a positive absolute temperature.

    The result is a Python scalar. Temperature validation belongs to the caller;
    this helper performs no device placement, sampling, or state update.

    Args:
        T__K: Finite positive absolute temperature.

    Returns:
        Thermal voltage as a Python float.
    """
    return thermal_fluctuation_energy__fJ(T__K) / ELEM_CHARGE__fC


# ### Charge ###


@overload
def q_conduction__fC(
    *,
    i__uA: float,
    duration__ns: float,
) -> float: ...


@overload
def q_conduction__fC(
    *,
    i__uA: Tensor,
    duration__ns: float,
) -> Tensor: ...


def q_conduction__fC(
    *,
    i__uA: Tensor | float,
    duration__ns: float,
) -> Tensor | float:
    """Charge transferred by a constant branch current."""
    return i__uA * duration__ns


@overload
def delta_q_cap__fC(
    *,
    c__fF: float,
    delta_v__V: float,
) -> float: ...


@overload
def delta_q_cap__fC(
    *,
    c__fF: float,
    delta_v__V: Tensor,
) -> Tensor: ...


@overload
def delta_q_cap__fC(
    *,
    c__fF: Tensor,
    delta_v__V: Tensor | float,
) -> Tensor: ...


def delta_q_cap__fC(
    *,
    c__fF: Tensor | float,
    delta_v__V: Tensor | float,
) -> Tensor | float:
    """Signed charge change of a constant capacitance.

    The result retains the voltage-change sign for nonnegative capacitance.
    """
    return c__fF * delta_v__V


# ### Supply energy ###


@overload
def e_charge__fJ(
    *,
    v_supply__V: float,
    delta_q_abs__fC: float,
) -> float: ...


@overload
def e_charge__fJ(
    *,
    v_supply__V: float,
    delta_q_abs__fC: Tensor,
) -> Tensor: ...


def e_charge__fJ(
    *,
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
    *,
    v_supply__V: float,
    c__fF: float,
    delta_v_abs__V: float,
) -> float: ...


@overload
def e_cap__fJ(
    *,
    v_supply__V: float,
    c__fF: Tensor,
    delta_v_abs__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap__fJ(
    *,
    v_supply__V: float,
    c__fF: float,
    delta_v_abs__V: Tensor,
) -> Tensor: ...


def e_cap__fJ(
    *,
    v_supply__V: float,
    c__fF: Tensor | float,
    delta_v_abs__V: Tensor | float,
) -> Tensor | float:
    """Supply energy consumed for a grounded capacitance's voltage change.

    The caller supplies nonnegative capacitance and absolute voltage change, and
    bills each excursion once. Inputs are used without validation or clamping.

    Args:
        v_supply__V: Driver supply potential, distinct from the node voltage.
        c__fF: Nonnegative capacitance, broadcastable with voltage changes.
        delta_v_abs__V: Nonnegative magnitude of the node voltage change.

    Returns:
        Supply energy under normal scalar or tensor broadcasting.
    """
    return v_supply__V * c__fF * delta_v_abs__V


@overload
def e_cap_excursion__fJ(
    *,
    v_supply__V: float,
    c__fF: float,
    v_rest__V: float,
    v_work__V: float,
) -> float: ...


@overload
def e_cap_excursion__fJ(
    *,
    v_supply__V: float,
    c__fF: Tensor,
    v_rest__V: Tensor | float,
    v_work__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap_excursion__fJ(
    *,
    v_supply__V: float,
    c__fF: float,
    v_rest__V: Tensor,
    v_work__V: Tensor | float,
) -> Tensor: ...


@overload
def e_cap_excursion__fJ(
    *,
    v_supply__V: float,
    c__fF: float,
    v_rest__V: float,
    v_work__V: Tensor,
) -> Tensor: ...


def e_cap_excursion__fJ(
    *,
    v_supply__V: float,
    c__fF: Tensor | float,
    v_rest__V: Tensor | float,
    v_work__V: Tensor | float,
) -> Tensor | float:
    """Supply energy for one rest-to-work-to-rest capacitive excursion.

    One call accounts for the complete excursion in either voltage direction.
    The caller supplies nonnegative capacitance and the supply potential behind
    that node.
    """
    delta_v__V = v_work__V - v_rest__V
    # Scalar versus tensor is fixed during tracing.
    delta_v_abs__V = delta_v__V.abs() if isinstance(delta_v__V, Tensor) else abs(delta_v__V)
    return e_cap__fJ(v_supply__V=v_supply__V, c__fF=c__fF, delta_v_abs__V=delta_v_abs__V)
