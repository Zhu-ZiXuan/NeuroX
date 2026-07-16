# RRAM

`RRAM` is a stateful conductance-cell module owning the programmed conductance buffer, the programming write, the per-call read snap, and the current and differential-conductance solve.

## Design decisions

- **`g_max__uS` is an `__init__` kwarg, not a config field.** The ceiling is a design value, not an intrinsic device parameter, so it stays outside the frozen `RRAMConfig` per the project rule that device design parameters are explicit constructor arguments. `g_min__uS` (intrinsic) lives in the config. The constructor enforces `g_max__uS > config.g_min__uS`.
- **Policy is separate from config and constructed per call site.** `RRAMConfig` carries the physics; `RRAMPolicy` carries the per-run decision of which non-idealities are live, as flat `bool` fields with no defaults so every call site states its intent explicitly. The policy is loaded from its own file alongside the config, never embedded in the device physics.
- **Program-time vs read-time sources are partitioned by surface.** Programming Gamma, drift, and stuck-at are baked into `g__uS` by `program(...)`; telegraph and thermal are resampled per `snapshot(...)`. This split is a contract, not an accident: re-running `snapshot` must not re-roll the programmed-in faults.

## Contracts & invariants

- **`program(...)` mutates state by buffer reassignment.** The stored conductance is replaced (`self.g__uS = ...`), not edited in place, so the buffer can grow from its scalar-zero initial shape to the programmed shape. Consumers must read `g__uS` fresh after a program, not cache a view.
- **`snapshot(shape, multi_coords)` is the only read path into fabricated state.** It expands `g__uS` to the per-call broadcast `shape`, optionally advanced-indexes a chunk via `multi_coords` (the chunked-solve selector), applies the read-time noise stack, and re-clamps. `multi_coords=None` returns the full broadcast view. Fabricated buffers stay on the device and are never mirrored into the snapshot caller.
- **`solve_dc(v, snap)` is stateless in the device.** It reads conductance only from the passed `RRAMSnap`, never from `self.g__uS`, so a chunk's snap and its solve stay paired.
- **No static fabricate mismatch.** `RRAM` joins the fabricate cascade but declares an explicit no-op `_sample_fabricate_mismatch` — RRAM variation enters through `program(...)` (state-dependent Gamma, stuck-at), not through `fabricate()`.

## Performance & resources

The state is one conductance buffer at the programmed broadcast shape. The read snap allocates noise draws at the per-call `shape` (or the indexed chunk); the chunked-solve `multi_coords` path sizes that allocation per chunk rather than over the full leading batch.

## Gotchas

- **Drift is unconditional, not policy-gated.** Unlike the four policy-flagged sources, the power-law drift gain is applied whenever `drift_decay_rate > 0` and `t_elapsed > drift_t0` — there is no `RRAMPolicy` switch for it. To disable drift, set the device parameters, not a policy flag.
- **`alpha == 0` is a distinct branch.** The linear I-V path returns `g.expand_as(i)` for the differential conductance; do not assume the `sinh`/`cosh` form is always taken. The branch is on the config value, so it is compile-time-constant per instance.
- **Read noise is reseeded every snapshot.** Two snaps of the same programmed state differ under a noise-on policy; chunked reads are therefore not bit-identical to a single-block read with noise on.

## Known limitations

- No dedicated device-level test module; `RRAM` is exercised through the 1T1R cell and physics paths (`tests/test_xbar_cell.py`, `tests/test_xbar_physics.py`). A focused device unit test (I-V, write fixed point, per-source noise statistics) is a coverage gap.

---

- **Reference**: [rram](../../../reference/primitive/device/rram.md)
- **Implementation**: `neurox/primitive/device/rram.py`
- **Tests**: `tests/test_xbar_cell.py`, `tests/test_xbar_physics.py`
