# analog_mux — Implementation

## Summary

`AnalogMux` (`analog_mux.py`) is a leaf differential transport block: it moves a grouped signal and tallies transport energy, with optional CM / DM noise applied per call. Spec: [reference/analog/analog_mux](../../reference/analog/analog_mux.md).

## Design decisions

- **Not polymorphic.** There is one concrete mux; parent circuits construct it directly from its config rather than dispatching through a family base. Adding a registry would buy nothing while a single topology exists, and the canonical leaf signature already standardises construction.
- **All noise is dynamic.** CM and DM noise are sampled inside `transport`, not fabricated; the mux owns no static mismatch, so `_sample_fabricate_mismatch` stays the inherited no-op.

## Contracts & invariants

- **Canonical leaf signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` matches the other leaves; the per-instance count is locked from `inst_shape` at construction.
- **Value-preserving.** The mux transports and aggregates only - it performs no weighting or arithmetic on the signal value (that lives in the composing circuit's switch-cap kernels). The two noise terms are the only departure from identity transport.

## Performance & resources

N/A - transport is a per-call elementwise map off the memory- and compile-critical path.

## Gotchas

- N/A.

## Known limitations

- N/A.

---

- **Reference**: [analog_mux](../../reference/analog/analog_mux.md)
- **Implementation**: `neurox/analog/analog_mux.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.
