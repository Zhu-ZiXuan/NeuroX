# Physics

`neurox/primitive/physics.py` holds the physical axioms every model is written from: the canonical SI physical constants — `ELEM_CHARGE__C`, `K_BOLTZMANN__J_per_K`, `EPS_0__F_per_m` — the standard reference temperature `T_ROOM__K`, and the closed-form laws over them, `thermal_voltage__V` and the capacitive supply-draw bill `e_cap_excursion__fJ`.

## Design decisions

- **Device- and circuit-level physical parameters come from config, never a hard-coded literal.** The constants here are the single exception, because they do not vary by device, process, or chip and so are fixed on purpose. `thermal_voltage__V` stays a helper derived from those constants rather than a stored value, so it adds no independent magic number of its own.
- **A stateless closed-form law lives with the constants, not in a module of its own.** A law that takes plain numbers, owns no configuration, and holds for every scheme is an axiom of the same kind as a constant, so the two share one file rather than one file per formula. `e_cap_excursion__fJ` is the whole of the bill every node-capacitance model applies; the physics it states is specified in [physics](../../reference/primitive/physics.md), and each caller cites that document instead of restating the law.
- **A shape-invariant law carries no shape annotation.** `e_cap_excursion__fJ` is elementwise over its capacitance and displacement arguments, so its result shape is whatever broadcasting gives and a `Shape:` line could only restate `[...]`. What the caller does need — that the capacitance must broadcast against the displacement — is stated as prose in the argument entry.

---

- **Reference**: [physics](../../reference/primitive/physics.md) — the constants and the supply-draw billing law.
- **Implementation**: `neurox/primitive/physics.py`
- **Tests**: `tests/primitive/test_physics.py`
