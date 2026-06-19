# driver — Implementation

## Summary

`Driver` (`driver.py`) is the ideal constant-voltage SL clamp-driver: one sampled clamp voltage, no feedback loop, no shape-dependent state. Spec: [reference/analog/driver](../../reference/analog/driver.md).

## Design decisions

- **Not a subclass of `TIA`, and no shared abstract base.** A `Driver` is an ideal voltage source; a `TIA` closes a feedback loop on a virtual ground - different physics, different parameter spaces. They share only the shape of the solver-facing clamp entry, which is too thin to justify an abstract base nothing else needs. The array solve binds the two as concrete types (`bl_driver: TIA`, `sl_driver: Driver`) rather than through a common interface. Rejected - a shared `ClampDriver` base: it would force a least-common-denominator config and obscure that the two clamps are physically unlike.
- **The clamp derivative is returned as a literal zero tensor.** The ideal voltage source has zero output impedance, so `solve_clamp` returns `dVclamp_dI__MOhm == 0` rather than omitting it; the array solve consumes a uniform `(v_clamp, dVclamp_dI)` pair from either boundary actor, so the zero must be present to keep the boundary interface shape-identical with the TIA.
- **No fabricated mismatch.** The only state is a scalar nominal drive voltage; there is no shape-dependent static mismatch, so `_sample_fabricate_mismatch` stays the inherited `FabricateMixin` no-op.

## Contracts & invariants

- **Solver-facing surface.** `v_ref__V` (the zero-current clamp voltage), `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)`, and `snapshot(*, shape, multi_coords)`. The richer `solve_dc(...)` is a concrete-handle extension the array solve does not depend on.
- **Construction commits `inst_shape`.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` locks the per-instance shape; the nominal drive value is a 0-d non-persistent buffer, so there is no shape-dependent fabricated buffer to rebuild.
- **Thermal noise is per-snapshot.** The optional Gaussian is sampled into the `DriverSnapshot`, not at construction; the same driver yields a fresh sample each read.

## Performance & resources

N/A - a scalar clamp with no inner solve and no shape-dependent state.

## Gotchas

- **The zero derivative is load-bearing.** Removing the `dVclamp_dI__MOhm` zero (because "an ideal source has no impedance") breaks the array solve's boundary interface, which expects the same return arity from both boundary actors.

## Known limitations

- N/A.

---

- **Reference**: [driver](../../reference/analog/driver.md)
- **Implementation**: `neurox/analog/driver.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.
