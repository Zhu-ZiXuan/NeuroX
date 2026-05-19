# `neurox/common/physical_constant.py`

## Current role

A small file of named physical constants and helpers, kept in `common/` so device / analog modules pull them from one canonical place.

## Exposed names

- `ELEM_CHARGE__C` — elementary charge [C].
- `K_BOLTZMANN__J_per_K` — Boltzmann constant [J/K].
- `EPS_0__F_per_m` — vacuum permittivity [F/m].
- `T_ROOM__K` — room temperature [K], the default operating temperature when no explicit `T__K` is supplied.
- `thermal_voltage__V(T__K)` — `k_B · T / q` evaluated at the given temperature [V].

## Naming convention

Constants follow NeuroX's project-wide `name__unit` style. Adding a new constant means adding a clear physical name + SI unit suffix; the unit is readable at the call site without consulting this file.
