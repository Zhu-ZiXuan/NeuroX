# TIA base

The abstract root the TIA family inherits: it carries the config-dispatched registry and `from_config`, the profiler registration, the `snapshot` / `solve_clamp` surface, and the snap type parameter `SnapT`. Concrete topologies live alongside; the reference clamp voltage is not owned by the family.

## Design decisions

- **The reference is injected, not a property or config field.** `snapshot` takes a keyword-only `v_ref__V: Tensor` that each concrete TIA stores in its snap, and `solve_clamp` reads the reference back from the snap. There is no `v_ref__V` property and no `v_ref__V` config field; the value is threaded in per call.
- **`TIAConfig` is the family's config base type.** It subclasses `CircuitConfig` and adds no field of its own — it exists to give the family a named config base type and to carry the static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) `CircuitConfig` defines for every circuit. Concrete configs subclass it and add their topology design parameters; per-op latency is leaf-defined, not a base field.
- **Snap type parameter.** `TIA` is `Generic[SnapT]` bound to `TIASnap`; each concrete TIA fixes its own snap dataclass and `snapshot()` / `solve_clamp()` carry that concrete type, so the abstract surface stays type-checked across topologies.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` — all six keyword-only and required. The base forwards `config / name / inst_shape` to `CircuitBase` (which binds `self.config` and `self._inst_shape` and registers the profiler) and discards `policy / dtype / T__K` for the concrete subclass to capture. `FabricateMixin` supplies the auto-cascade `fabricate()`; concrete TIAs override `_sample_fabricate_mismatch`.
- **Clamp-solve surface.** `snapshot(*, v_ref__V, shape, multi_coords) -> SnapT` (the reference injected here, stored in the snap) plus `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` — nothing else; the abstract surface carries no concrete extension.
- **Empty marker `TIAPolicy`.** The base policy carries no field; each concrete TIA declares its own `*Policy(TIAPolicy)`.

---

- **Reference**: [tia base](../../../../reference/primitive/analog/tia/family.md)
- **Implementation**: `neurox/analog/tia/base.py`
- **Tests**: TODO - name the guarding test
