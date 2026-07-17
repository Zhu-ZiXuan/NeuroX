# Current subtractor

## Design decisions

- **Not polymorphic.** A single concrete class, not a scheme family; its `Policy` carries two toggles, `mismatch` and `offset`.
- **Magnitude-and-sign value path.** `subtract` forms one signed difference `delta = i_a - ratio_mismatch * i_b + offset` and returns the gained magnitude `gain * delta.abs()` with the boolean direction bit `sign = delta < 0`; it logs no dynamic energy — it is a current-domain primitive, and the downstream consumer that owns the rail tallies dissipation. With both toggles off the difference is exact (`ratio_mismatch` unit, `offset__uA` zero); with them on, each is a static per-instance factor sampled once at `fabricate` into a held non-persistent buffer — `ratio_mismatch` a `1 + randn(inst_shape) * mismatch_sigma_relative` multiplier, `offset__uA` an additive `randn(inst_shape) * offset_sigma__uA` — then broadcast over the leading batch / element dims of the inputs. Sampling into the buffers — never a per-call draw on the inputs — fixes both as fabrication properties held across reads.
- **One signed difference feeds both outputs.** The magnitude and the sign read the same `delta`, so the input-referred `offset__uA` perturbs both; this is the intended input-referred model. `sign` is returned as a bool tensor, leaving the cast to the consumer's own code dtype.
- **Both statics via the shared Gaussian.** `_sample_fabricate_mismatch` draws each buffer with [nonideality](../nonideality.md)'s `apply_gaussian` off a `ones` / `zeros` template under the matching `enabled=` toggle, so a disabled source returns the identity template unchanged.

## Contracts & invariants

- **Canonical leaf signature.** The per-instance count is locked from `inst_shape` at construction, and both mismatch buffers are shaped `inst_shape`.
- **Static PPA rolls up to the owner.** The subtractor accounts no dynamic energy, and its silicon is accounted in the owning current-domain circuit's config, so it declares no per-instance area or leakage data of its own and sets `is_profile_target` false ([base](base.md)).

## Performance & resources

N/A — `subtract` is a per-call elementwise map off the memory- and compile-critical path.

## Known limitations

- Output compliance, finite output impedance, and rail-steering headroom are not modelled.
- A single input-referred offset drives both the magnitude and the sign; separate magnitude-path gain error and sign-path offset are not modelled independently.

---

- **Reference**: [current_subtractor](../../../reference/primitive/analog/current_subtractor.md)
- **Implementation**: `neurox/primitive/analog/current_subtractor.py`
- **Tests**: `tests/primitive/analog/test_current_subtractor.py`
