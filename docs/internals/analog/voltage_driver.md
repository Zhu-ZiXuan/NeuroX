# Voltage driver

## Summary

`VoltageDriver` (`voltage_driver.py`) is the generic Thevenin voltage-source clamp: a reference voltage `v_ref` behind a constant series resistance `r_out`, with a closed-form clamp solve and no inner loop. It is the ideal constant-voltage source in the `r_out = 0` limit.

## Design decisions

- **Satisfies the `ClampDriver` role structurally, no inheritance.** `VoltageDriver` exposes `snapshot(*, v_ref__V, shape, multi_coords)` and `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` with signatures matching the structural `ClampDriver` `Protocol`, but does not import or subclass it; conformance is structural (by signature), not nominal. The reference rides in the snap, injected through `snapshot`'s `v_ref__V` keyword; the role exposes no standalone `v_ref__V` member.
- **No conduction energy, hence no `dynamic_energy*` method.** The conduction power that holds the clamp under load is burned in the supply rail the consumer owns, not in this block; tallying it here would double-count it against the rail owner, so the block reports only its own static power via the inherited `leakage_per_inst__uW` (which carries any internal amplifier / bias). There is correspondingly no `v_dd` field and no bias-current field — the source is purely behavioural.
- **The clamp slope is returned as a real constant, not a zero.** `solve_clamp` returns `dVclamp_dI__MOhm == r_out` (`expand_as` the port current); at `r_out = 0` this is a literal zero, and a finite `r_out` is the physical series impedance that serves as the clamp slope. The pair is always present to keep the boundary interface shape-identical with the other boundary actor.
- **No `solve_dc` handle.** `VoltageDriver` exposes only the tuple-based `solve_clamp`; the closed-form affine clamp needs no DCOP container, so there is no `solve_dc` / `*DCOP` on this block.

## Contracts & invariants

- **Construction commits `inst_shape`.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` is the canonical leaf signature. The frozen `r_out` is a 0-d non-persistent buffer; the systematic per-instance `offset__V` is a second non-persistent buffer at `inst_shape`, sampled at `fabricate`. There is no nominal-`v_ref` buffer — the reference is injected per call, not held.
- **The reference is injected, not config-held.** `VoltageDriver` carries no `v_ref__V` config field and no `v_ref__V` property; `snapshot(*, v_ref__V, shape, multi_coords)` takes the reference as a `Tensor` keyword, applies offset then thermal to it, and stores the result in the snap.
- **Closed-form `solve_clamp`.** `v_clamp = snap.v_ref__V - i_port__uA * snap.r_out__MOhm` has no data-dependent control flow, so it introduces no graph break under `torch.compile`. `v_clamp_init__V` is accepted (interface parity with iterative clamps) and ignored.
- **Snap carries both fields.** `VoltageDriverSnap` holds the noised `v_ref__V` (broadcast to the per-call shape) and the 0-d frozen `r_out__MOhm`, so `solve_clamp` is a pure function of the snap.

## Performance & resources

N/A — an affine clamp with no inner solve and no shape-dependent state.

## Gotchas

- **Static offset then per-call thermal, on the injected reference.** `snapshot` adds the held per-instance offset to a cloned view of the injected `v_ref__V`, then the per-call `thermal_sigma__V` via `apply_gaussian`; the offset is fixed across reads while the thermal term is fresh each read.
- **`multi_coords` chunk-selects on the broadcast view.** The injected reference and the held offset are both `expand`ed to `shape` and indexed by `multi_coords` in lockstep before the thermal draw; `r_out` is the shared 0-d buffer and is not chunk-indexed.

## Known limitations

- N/A.

---

- **Reference**: [voltage_driver](../../reference/analog/voltage_driver.md)
- **Implementation**: `neurox/analog/voltage_driver.py`
- **Tests**: TODO — name the guarding test
