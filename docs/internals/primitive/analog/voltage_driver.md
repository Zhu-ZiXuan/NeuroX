# Voltage driver

## Design decisions

- **`drive` is the billing seam, and it bills from the converged port state.** The clamp's dynamic energy is emitted by `drive(i_port__uA, v_clamp__V)`, called once per access after the port has settled, not by `snapshot` and not by `solve_dc`. `snapshot` samples state before the solve knows any current, and `solve_dc` runs once per solver iteration, so billing at either would bill an unsettled or a repeated operating point. `drive` returns the clamp voltage it was handed: the solved node is already the delivered one, so a caller that reads the return and a caller that keeps the solved node see the same value. It is an ordinary method of this concrete block, not part of the structural clamp role the solve consumes, so its caller is whoever owns this driver.
- **The event is a flat per-op lump, so it is emitted as a 0-dim constant expanded onto the port state's layout.** `energy_per_op__fJ` is constant per port operation (the Reference accounting boundary), so the expanded view holds no storage and nothing is materialized over the port state; `i_port__uA` fixes the layout only, and the energy dtype is the constant's. The timed conduction share, `i_port**2 * r_out` included, is not this block's: it needs a conduction window the clamp has no notion of. There is consequently no conduction-energy method, no `v_dd` field, and no bias-current field, and static power rides `leakage_per_inst__uW` on its own config.
- **The port axis is summed with the rest, undeclared.** `drive` emits at `i_port__uA`'s layout; the collector keeps `i_port__uA.shape[:leading_rank]` — the caller's own leading dims — and sums every axis past it, including this driver's own per-port instance axis. Nothing about the port axis is declared at the emission site; a single-instance clamp and a multi-instance one both reduce the same way.
- **The clamp slope is returned as a real expanded constant, not a special-cased zero.** `solve_dc` fills `dvclamp_di__MOhm` with `-snap.r_out__MOhm.expand_as(i_port__uA)` — the derivative of the Thevenin map, which the series drop makes the NEGATED resistance, broadcast to the port current, a literal zero tensor at `r_out = 0`, never a scalar special case. The field's shape is invariant across output-resistance values.
- **The solve returns a DCOP container, closed form or not.** `solve_dc` returns `VoltageDriverDcop(v_clamp__V, dvclamp_di__MOhm)` rather than a tuple, which is what the structural `ClampDcop` bound reads; the container costs this affine clamp nothing and keeps a driver free to publish its own extra outputs beside the two the role requires.

## Contracts & invariants

- **Construction commits `inst_shape`.** `__init__(*, config, policy, inst_shape, dtype, T__K)` is the canonical leaf signature. `_frozen_r_out__MOhm` is a 0-D circuit constant buffer and `_nominal_offset__V` is a 0-D nominal buffer. `fabricate()` expands the latter to `inst_shape` and assigns `_offset__V` ordinary state. There is no nominal-`v_ref` buffer; the reference is injected per call, not held.
- **The reference is injected, not config-held.** `VoltageDriver` carries no `v_ref__V` config field and no `v_ref__V` property; `snapshot(*, v_ref__V, shape)` takes the reference as a `Tensor` keyword and keeps it in the snap unchanged.
- **`shape` is an explicit argument, and it is the event shape.** The reference source it is handed is a static identity that says nothing about how many accesses this call covers, so the shape has to be stated rather than read off the reference. The caller — the macro that schedules the accesses — is the only party that knows it. Everything the driver samples is materialized at that shape: the reference expands onto it, the fabricated offset expands onto it, and the thermal draw takes one fresh sample per position of it.
- **`drive` is called once per access, on the full-shape port state.** Its two arguments are laid out `[*caller_leading, *middle, *inst_shape]` with the port axis last and nothing trailing it. Every quantity the snap carried — offset and thermal draw alike — is already inside the solved clamp voltage, so `drive` needs no snap.
- **Closed-form `solve_dc`.** `v_clamp = snap.v_ref__V + snap.v_perturb__V - i_port__uA * snap.r_out__MOhm` has no data-dependent control flow, so it introduces no graph break under `torch.compile`. `v_clamp_init__V` is accepted (interface parity with iterative clamps) and ignored.
- **Snap carries three fields, and the perturbation is separate on purpose.** `VoltageDriverSnap` holds the NOMINAL `v_ref__V`, this driver's own `v_perturb__V`, and the frozen `r_out__MOhm`, so `solve_dc` is a pure function of the snap. Keeping the perturbation out of `v_ref__V` is what lets a consumer read the ideal level off the snap — the array's capacitive billing measures displacement from it, and the solver seeds its Newton from it — while `solve_dc` still delivers the perturbed clamp. Under an all-off policy `v_perturb__V` is exactly zero, so the clamp reduces to the nominal reference bit-exactly. All three arrive at the per-call shape: the constant slope is expanded to it rather than left 0-d, so a snap states one value per driven position and no consumer has to special-case a field that carries no shape. The expand is a stride-0 view, so the constant still costs one element.

## Performance & resources

N/A — no inner solve and no shape-dependent state.

## Gotchas

- **Static offset then per-call thermal, accumulated in `v_perturb__V`.** `snapshot` expands the held per-instance offset onto `shape`, then adds the per-call `thermal_sigma__V` draw via `apply_gaussian`; the offset is fixed across reads while the thermal term is fresh each read. Neither touches `v_ref__V`.
- **The port state must carry the caller's leading dims at their true extents.** `drive`'s only shape requirement is that `i_port__uA` opens with the caller's own leading dims; a size-1 stand-in for a real caller extent is not a valid broadcast, and the event is then billed as the single unit operation it claims to be.
- **`shape` fixes the noise extent.** The thermal draw covers exactly the positions `shape` spans, so a `shape` short of the real event count takes a narrower draw and shares one sample across accesses that are physically distinct. `_frozen_r_out__MOhm` stays the shared 0-D buffer, expanded into the snap as a stride-0 view.

## Known limitations

- N/A.

---

- **Reference**: [voltage_driver](../../../reference/primitive/analog/voltage_driver.md)
- **Implementation**: `neurox/primitive/analog/voltage_driver.py`
- **Tests**: TODO — name the guarding test
