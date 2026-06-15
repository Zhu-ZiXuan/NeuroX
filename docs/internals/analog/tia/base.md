# TIA base — Implementation

## Summary

The TIA family: the abstract `TIA` (`tia/base.py`) carrying the registry, `from_config`, profiler registration, the abstract `v_ref__V` property, and the snapshot type parameter. Concrete impls live alongside ([opamp_tia](opamp_tia.md)). Spec: [reference/analog/tia/base](../../../reference/analog/tia/base.md).

## Design decisions

- **`v_ref__V` is an abstract property, not base-stored.** The base owns no `config` storage, so it cannot implement the accessor; every concrete TIA returns it from its own `self.config` (typically `self.config.v_ref__V`). This is the one solver-facing contract every topology must honour explicitly, so it is `@property @abstractmethod`.
- **`TIAConfig` is a real base config, not an empty marker.** Unlike the ADC / DAC marker bases, the TIA base config holds the orchestration-level fields every topology shares (`v_ref__V`, leakage / area / latency); concrete configs inherit and add topology design parameters. The shared fields are genuinely common across TIA topologies, so they belong on the base.
- **Snapshot type parameter.** `TIA` is `Generic[SnapshotT]` bound to `TIASnapshot`; each concrete TIA fixes its own snapshot dataclass and `snapshot()` / `solve_clamp()` carry that concrete type, so the solver-facing surface stays type-checked across topologies.
- **No shared base with `Driver`.** The BL clamp (TIA) and the SL clamp (`Driver`) are physically unlike; the solver binds them as concrete types. See [driver internals](../driver.md) for the rejected-shared-base rationale.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `config / policy / dtype / T__K`; concrete subclasses store the rest. `FabricateMixin` provides the auto-cascade `fabricate()`; concrete TIAs override `_sample_fabricate_mismatch`.
- **Solver-facing surface.** `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` plus the `v_ref__V` property - nothing else. The array solve binds the BL boundary actor to this surface abstractly; it does not depend on any concrete extension.
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
- **Decisions**: N/A — no ADR governs this module.
