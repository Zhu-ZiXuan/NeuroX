"""Physical constants used across NeuroX device, circuit, and ADC models.

All values are CODATA 2018 (or definitionally exact post-2019 SI
redefinition) and carry their SI unit in the name suffix.  Reuse
these constants instead of hard-coding magic numbers — keeps the
device, NMOS, RRAM, and ADC kT/C noise paths consistent and makes
unit-test parity trivial.

Examples:
    >>> from neurox.common.physical_constant import K_BOLTZMANN__J_per_K, ELEM_CHARGE__C
    >>> V_T_at_300K = K_BOLTZMANN__J_per_K * 300.0 / ELEM_CHARGE__C  # ≈ 0.02585 V
"""

from __future__ import annotations

# Boltzmann constant (exact post-2019 SI redefinition).
K_BOLTZMANN__J_per_K: float = 1.380649e-23

# Elementary charge (exact post-2019 SI redefinition).
ELEM_CHARGE__C: float = 1.602176634e-19

# Vacuum permittivity (CODATA 2018).
EPS_0__F_per_m: float = 8.8541878128e-12

# Standard reference temperature (room temperature).
T_ROOM__K: float = 300.0


def thermal_voltage__V(temperature__K: float) -> float:
    """Thermal voltage ``V_T = k_B · T / q`` in [V].

    At ``T = 300 K`` this evaluates to ≈ 0.02585 V — the standard
    value used in subthreshold MOSFET and kT/C-noise models.

    Args:
        temperature__K: Absolute temperature in Kelvin.  Must be > 0.

    Returns:
        Thermal voltage in volts.
    """
    if temperature__K <= 0.0:
        raise ValueError(f"temperature__K ({temperature__K}) must be > 0")
    return K_BOLTZMANN__J_per_K * temperature__K / ELEM_CHARGE__C


__all__ = [
    "ELEM_CHARGE__C",
    "EPS_0__F_per_m",
    "K_BOLTZMANN__J_per_K",
    "T_ROOM__K",
    "thermal_voltage__V",
]
