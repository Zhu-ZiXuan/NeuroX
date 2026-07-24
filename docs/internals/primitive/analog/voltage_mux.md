# Voltage mux

## Design decisions

- **Not polymorphic.** There is one concrete mux; parent circuits construct it directly from its config rather than dispatching through a family base.
- **Static mismatch vs dynamic noise.** `_sample_fabricate_mismatch` samples inter-leg gain mismatch `_eps_g` once at fabricate from `mux_gain_mismatch_sigma_relative`; `transport` then applies the resulting per-leg static gains, while CM and DM noise stay dynamic, re-sampled inside `transport`.

## Contracts & invariants

- **Canonical leaf signature.** `__init__(*, config, policy, inst_shape, dtype, T__K)` matches the other leaves; the per-instance count is locked from `inst_shape` at construction.
- **Per-call PPA tally.** `transport` logs per-access dynamic energy and a latency of the per-op cost times the serial-op count; with no parallel trailing beyond `inst_shape`, that count is `numel // inst_count`.

## Performance & resources

N/A — transport is a per-call elementwise map off the memory- and compile-critical path.

---

- **Reference**: [voltage_mux](../../../reference/primitive/analog/voltage_mux.md)
- **Implementation**: `neurox/primitive/analog/voltage_mux.py`
- **Tests**: TODO — name the guarding test
