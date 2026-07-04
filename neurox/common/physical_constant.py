"""Physical constants in SI units (CODATA 2018).

See also:
    docs/internals/common/physical_constant.md
"""

from __future__ import annotations

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
        Thermal voltage in plain SI volts (not millivolts), matching the ``__V`` suffix.

    Raises:
        ValueError: If ``temperature__K`` is not positive.
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
