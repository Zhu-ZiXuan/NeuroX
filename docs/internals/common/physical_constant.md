# Physical constants

## Summary

`neurox/common/physical_constant.py` is the single source of the canonical physical constants shared across device and analog modules — `ELEM_CHARGE__C`, `K_BOLTZMANN__J_per_K`, `EPS_0__F_per_m`, and `T_ROOM__K` — plus the scalar helper `thermal_voltage__V(temperature__K)`. Their numeric values, symbols, per-constant meanings, the thermal-voltage definition, and the `name__unit` suffix grammar are the shared spec in [notation_conventions](../../conventions/notation_conventions.md).

## Design decisions

- **One canonical source.** The constants are defined once so every device and analog module imports them rather than embedding its own literal, which prevents drift between callers.
- **Thermal voltage is a helper, not a stored constant.** `thermal_voltage__V(temperature__K)` computes the value at the caller's temperature instead of exporting a fixed `V_T`, so a module running off the default temperature never reads a baked-in figure; temperature flows through the same `T__K` argument as the rest of the construction surface.

## Contracts & invariants

- **`thermal_voltage__V` requires a positive temperature.** A non-positive `temperature__K` raises `ValueError` rather than returning a degenerate or negative voltage.
- **The constants are config-side plain floats.** They are plain floats in the SI unit named by their suffix, intended for config-side derivation and per-module `__init__` conversion, not a tensor-computation path.

## Performance & resources

N/A — module-level float constants and one scalar arithmetic helper; no tensors, buffers, or allocations.

## Gotchas

- **The returned voltage is plain SI volts, not a runtime-unit tensor.** Do not feed it directly onto a tensor-computation path that expects the closed runtime-unit set; convert it inside the consuming class's `__init__` like any other config-unit datum.

## Known limitations

N/A — the constant set covers the quantities currently consumed.

---

- **Reference**: [notation_conventions](../../conventions/notation_conventions.md) — physical constants, symbols, unit-suffix grammar, and the thermal-voltage definition.
- **Implementation**: `neurox/common/physical_constant.py`
- **Tests**: TODO — no dedicated test module.
- **Decisions**: N/A.
