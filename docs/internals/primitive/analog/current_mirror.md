# Current mirror

## Design decisions

- **Not polymorphic.** A single concrete class, not a scheme family; its `Policy` carries the one `mismatch` toggle.
- **Pure ratio-copy value path.** `replicate` returns the ratio copy `mirror_ratio * ratio_mismatch * i_in` and logs no dynamic energy — it is a current-copy primitive, and the downstream current-domain consumer that owns the rail tallies dissipation. With `mismatch` off the multiplier is unit (the exact ratio); with it on, `ratio_mismatch` is a static per-instance factor `1 + randn(inst_shape) * ratio_sigma_relative` sampled once at `fabricate` into the held non-persistent `ratio_mismatch` buffer, then broadcast over the leading batch / im2col / element dims of `i_in`. Sampling into the buffer — never a per-call `randn_like` on the input — fixes the mismatch as a fabrication property held across reads.

## Contracts & invariants

- **Canonical leaf signature.** The per-instance count is locked from `inst_shape` at construction.
- **Static PPA rolls up to the owner.** The mirror's silicon is accounted in the owning current-domain circuit's config, so it declares no per-instance area or leakage data of its own and sets `reports_static_ppa` false ([base](base.md)).

## Performance & resources

N/A — `replicate` is a per-call elementwise map off the memory- and compile-critical path.

## Known limitations

- Output compliance, finite output impedance, and input offset are not modelled.

---

- **Reference**: [current_mirror](../../../reference/primitive/analog/current_mirror.md)
- **Implementation**: `neurox/primitive/analog/current_mirror.py`
- **Tests**: `tests/test_current_readout_energy.py`
