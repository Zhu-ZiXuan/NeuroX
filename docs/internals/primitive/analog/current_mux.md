# Current mux

## Design decisions

- **Not polymorphic.** One concrete ideal mux, constructed directly rather than dispatched through a registry. It has no error sources, so its `Policy` is an empty marker.
- **Ideal, lossless value path — no emission.** The value path applies only the matched gain — no numeric non-ideality — and `transport` self-logs neither rail energy nor latency, so a lossless copy contributes nothing to the dynamic tally. Governing equations in Reference.
- **`select_num` is design fan-in.** It sizes the modelled N:1 mux but is not a runtime multiplier of any tally.

## Contracts & invariants

- **Canonical leaf signature.** The per-instance count is fixed from `inst_shape` at construction.
- **Static PPA rolls up to the owner.** The mux's silicon is accounted in the owning current-domain circuit's config, so it declares no per-instance area or leakage data of its own and sets `is_profile_target` false ([base](base.md)).

## Performance & resources

N/A — per-call elementwise map off the memory- and compile-critical path.

## Known limitations

- Ideal only: no policy hook exists to enable the non-idealities named in Reference.

---

- **Reference**: [current_mux](../../../reference/primitive/analog/current_mux.md)
- **Implementation**: `neurox/primitive/analog/current_mux.py`
- **Tests**: `tests/test_current_readout_energy.py`
