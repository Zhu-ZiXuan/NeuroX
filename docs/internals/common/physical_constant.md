# Physical Constants — Implementation

## Summary

`neurox/common/physical_constant.py` is the single source of the canonical physical constants used across device and analog modules: `ELEM_CHARGE__C` (elementary charge), `K_BOLTZMANN__J_per_K` (Boltzmann constant), `EPS_0__F_per_m` (vacuum permittivity), and `T_ROOM__K` (room temperature, the default operating point when no explicit `T__K` is supplied). It also exposes `thermal_voltage__V(temperature__K)`, which derives the thermal voltage at a given temperature from the first two constants. The numeric values and their symbols live in the [notation conventions](../../reference/notation_conventions.md) reference and are not restated here; this document covers only the module contract.

## Design decisions

- **One canonical source.** The constants are defined once in this module so every device and analog module imports them rather than embedding its own literal. A single definition prevents drift between callers and makes the unit suffix (`__C`, `__J_per_K`, `__F_per_m`, `__K`) the only place a value carries its SI dimension.
- **Thermal voltage is derived, not stored.** There is no `V_T` constant. `thermal_voltage__V(temperature__K)` computes the value at the caller's temperature, so a module operating off the default room temperature does not silently read a baked-in 300 K figure. Temperature flows through the same `T__K` argument the rest of the construction surface uses.
- **`name__unit` naming.** Each name carries its SI unit suffix, so the dimension is readable at the call site without consulting this file. Adding a constant means adding a clear physical name plus its unit suffix.

## Contracts & invariants

- **`thermal_voltage__V(temperature__K)` requires a positive temperature.** The temperature must be strictly greater than zero; a non-positive argument raises `ValueError` rather than returning a degenerate or negative voltage.
- **Constants are SI-unit module-level values.** They are plain floats in the unit named by their suffix, intended for use in config-side derivations and per-module `__init__` conversions, not on a tensor-computation path (see the runtime-unit rule in the reference).
- **`T_ROOM__K` is the conventional default temperature.** By convention a module that accepts an optional operating temperature uses `T_ROOM__K` when none is supplied; this module exports the value but does not enforce the convention — the default lives at each consumer's construction surface.

## Performance & resources

N/A — module-level float constants and one scalar arithmetic helper; no tensors, buffers, or allocations.

## Gotchas

- **The returned voltage is in SI volts at the suffix unit, not a runtime-unit tensor.** Do not feed it directly onto a tensor-computation path that expects the closed runtime-unit set; convert inside the consuming class's `__init__` as with any config-unit datum.
- **`thermal_voltage__V` is a function, not a constant.** Because it depends on temperature, there is no module-level `V_T` to import; call the function at the operating temperature.

## Known limitations

- N/A — the constant set covers the quantities currently consumed; new constants are added by the `name__unit` rule above as modules require them.

---

- **Reference**: [notation_conventions § Physical constants](../../reference/notation_conventions.md)
- **Implementation**: `neurox/common/physical_constant.py`
- **Tests**: `tests/test_xbar_physics.py`
- **Decisions**: N/A.
