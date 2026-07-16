# Physical constants

## Summary

`neurox/primitive/physical_constant.py` holds the canonical SI physical constants — `ELEM_CHARGE__C`, `K_BOLTZMANN__J_per_K`, `EPS_0__F_per_m` — together with the standard reference temperature `T_ROOM__K` and the derived scalar helper `thermal_voltage__V`.

## Design decisions

- **Device- and circuit-level physical parameters come from config, never a hard-coded literal.** The constants here are the single exception, because they do not vary by device, process, or chip and so are fixed on purpose. `thermal_voltage__V` stays a helper derived from those constants rather than a stored value, so it adds no independent magic number of its own.

---

- **Reference**: N/A — software utility
- **Implementation**: `neurox/primitive/physical_constant.py`
- **Tests**: TODO — no dedicated test module.
