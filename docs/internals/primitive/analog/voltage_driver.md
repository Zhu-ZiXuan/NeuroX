# Voltage driver

## Design decisions

- **Satisfies the `ClampDriver` role structurally, no inheritance.** `VoltageDriver` exposes `snapshot(*, v_ref__V, shape, multi_coords)` and `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` with signatures matching the structural `ClampDriver` `Protocol`, but does not import or subclass it; conformance is structural (by signature), not nominal. The reference rides in the snap, injected through `snapshot`'s `v_ref__V` keyword; the role exposes no standalone `v_ref__V` member.
- **No conduction energy, hence no `dynamic_energy*` method.** The model tallies no conduction energy in this block — the accounting boundary is fixed in Reference — so it exposes no `dynamic_energy*` method, no `v_dd` field, and no bias-current field, and reports only its own static power through the inherited `leakage_per_inst__uW`.
- **The clamp slope is returned as a real expanded constant, not a special-cased zero.** `solve_clamp` returns `dVclamp_dI__MOhm` as `snap.r_out__MOhm.expand_as(i_port__uA)` — a tensor broadcast to the port current, a literal zero tensor at `r_out = 0`, never a scalar special case. The `(v_clamp__V, dVclamp_dI__MOhm)` pair is always returned so the boundary interface stays shape-identical with the other boundary actor.
- **No `solve_dc` handle.** `VoltageDriver` exposes only the tuple-based `solve_clamp`; the closed-form affine clamp needs no DCOP container, so there is no `solve_dc` / `*DCOP` on this block.

## Contracts & invariants

- **Construction commits `inst_shape`.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` is the canonical leaf signature. `frozen_r_out__MOhm` is a 0-d non-persistent buffer; the systematic per-instance `offset__V` is a second non-persistent buffer at `inst_shape`, sampled at `fabricate`. There is no nominal-`v_ref` buffer — the reference is injected per call, not held.
- **The reference is injected, not config-held.** `VoltageDriver` carries no `v_ref__V` config field and no `v_ref__V` property; `snapshot(*, v_ref__V, shape, multi_coords)` takes the reference as a `Tensor` keyword, applies offset then thermal to it, and stores the result in the snap.
- **Closed-form `solve_clamp`.** `v_clamp = snap.v_ref__V - i_port__uA * snap.r_out__MOhm` has no data-dependent control flow, so it introduces no graph break under `torch.compile`. `v_clamp_init__V` is accepted (interface parity with iterative clamps) and ignored.
- **Snap carries both fields.** `VoltageDriverSnap` holds the noised `v_ref__V` (broadcast to the per-call shape) and the 0-d frozen `r_out__MOhm`, so `solve_clamp` is a pure function of the snap.

## Performance & resources

N/A — no inner solve and no shape-dependent state.

## Gotchas

- **Static offset then per-call thermal, on the injected reference.** `snapshot` adds the held per-instance offset to a cloned view of the injected `v_ref__V`, then the per-call `thermal_sigma__V` via `apply_gaussian`; the offset is fixed across reads while the thermal term is fresh each read.
- **`multi_coords` chunk-selects on the broadcast view.** The injected reference and the held offset are both `expand`ed to `shape` and indexed by `multi_coords` in lockstep before the thermal draw; `frozen_r_out__MOhm` is the shared 0-d buffer and is not chunk-indexed.

## Known limitations

- N/A.

---

- **Reference**: [voltage_driver](../../../reference/primitive/analog/voltage_driver.md)
- **Implementation**: `neurox/primitive/analog/voltage_driver.py`
- **Tests**: TODO — name the guarding test
