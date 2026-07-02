# TIA base

## Summary

The TIA family: the abstract `TIA` (`tia/base.py`) carrying the registry, `from_config`, profiler registration, the abstract `snapshot` / `solve_clamp` surface, and the snap type parameter. The reference is injected per call into `snapshot` and carried in the snap, so there is no `v_ref__V` property on the family. Concrete impls live alongside ([opamp_tia](opamp_tia.md)). Spec: [reference/analog/tia/base](../../../reference/analog/tia/base.md).

## Design decisions

- **The reference is injected, not a property or config field.** `snapshot` gains a keyword-only `v_ref__V: Tensor` param that each concrete TIA stores in its snap; `solve_clamp` / `solve_dc` read the reference from the snap. There is no `v_ref__V` property and no `v_ref__V` config field — the consumer sources the value from a [voltage_reference](../voltage_reference.md) and threads the chosen tap in per call.
- **`TIAConfig` is a real base config, not an empty marker.** Unlike the ADC / DAC marker bases, the TIA base config holds the orchestration-level fields every topology shares (leakage / area / latency); concrete configs inherit and add topology design parameters. These shared static-PPA fields are genuinely common across TIA topologies, so they belong on the base.
- **Snap type parameter.** `TIA` is `Generic[SnapT]` bound to `TIASnap`; each concrete TIA fixes its own snap dataclass and `snapshot()` / `solve_clamp()` carry that concrete type, so the solver-facing surface stays type-checked across topologies.
- **No shared base with `VoltageDriver`.** The BL clamp (TIA) and the SL clamp (`VoltageDriver`) are physically unlike; the consuming array solve consumes both through the structural `ClampDriver` role per call, not as stored concrete types. That thin shared surface is named as the role, not a base class — see [voltage_driver internals](../voltage_driver.md) for the rejected-shared-base rationale.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `config / policy / dtype / T__K`; concrete subclasses store the rest. `FabricateMixin` provides the auto-cascade `fabricate()`; concrete TIAs override `_sample_fabricate_mismatch`.
- **Solver-facing surface.** `snapshot(*, v_ref__V, shape, multi_coords) -> SnapT` (the reference injected here, stored in the snap) plus `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` - nothing else. The array solve binds the BL boundary actor to this surface abstractly; it does not depend on any concrete extension.
- **Empty marker `TIAPolicy`.** Concrete TIAs declare their own `*Policy(TIAPolicy)`; the composite stores the abstract field type and the caller passes the concrete impl.

## Performance & resources

N/A at this level - the clamp evaluation cost is topology-specific.

## Gotchas

- N/A.

## Known limitations

- N/A.

---

- **Reference**: [tia base](../../../reference/analog/tia/base.md)
- **Implementation**: `neurox/analog/tia/base.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md)
