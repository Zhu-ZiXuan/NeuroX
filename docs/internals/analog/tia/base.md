# TIA base

## Summary

The TIA family: the abstract `TIA` (`tia/base.py`) carrying the registry, `from_config`, profiler registration, the abstract `snapshot` / `solve_clamp` surface, and the snap type parameter. The reference is injected per call into `snapshot` and carried in the snap, so there is no `v_ref__V` property on the family. Concrete impls live alongside.

## Design decisions

- **The reference is injected, not a property or config field.** `snapshot` gains a keyword-only `v_ref__V: Tensor` param that each concrete TIA stores in its snap; `solve_clamp` / `solve_dc` read the reference from the snap. There is no `v_ref__V` property and no `v_ref__V` config field — the consumer sources the value from a [voltage_reference](../voltage_reference.md) and threads the chosen tap in per call.
- **`TIAConfig` is the shared base config.** Like the ADC / DAC config bases, it subclasses `CircuitConfig` and adds no field of its own: it gives the parent config's TIA field a base type to name and carries the static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) `CircuitConfig` defines for every circuit. Concrete configs subclass it and add their topology design parameters; per-op latency is leaf-defined, not a base field.
- **Snap type parameter.** `TIA` is `Generic[SnapT]` bound to `TIASnap`; each concrete TIA fixes its own snap dataclass and `snapshot()` / `solve_clamp()` carry that concrete type, so the solver-facing surface stays type-checked across topologies.
- **No shared base with `VoltageDriver`.** The BL clamp (TIA) and the SL clamp (`VoltageDriver`) are physically unlike; the consuming array solve consumes both through the structural `ClampDriver` role per call, not as stored concrete types. That thin shared surface is named as the role, not a base class — the rejected-shared-base rationale is stated with the SL clamp.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `config / policy / dtype / T__K`; concrete subclasses store the rest. `FabricateMixin` provides the auto-cascade `fabricate()`; concrete TIAs override `_sample_fabricate_mismatch`.
- **Solver-facing surface.** `snapshot(*, v_ref__V, shape, multi_coords) -> SnapT` (the reference injected here, stored in the snap) plus `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` - nothing else. A consumer binds to this surface abstractly and depends on no concrete extension.
- **Empty marker `TIAPolicy`.** Concrete TIAs declare their own `*Policy(TIAPolicy)`; the composite stores the abstract field type and the caller passes the concrete impl.

---

- **Reference**: [tia base](../../../reference/analog/tia/family.md)
- **Implementation**: `neurox/analog/tia/base.py`
- **Tests**: TODO - name the guarding test
