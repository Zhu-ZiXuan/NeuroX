# RRAM

`Rram` is a stateful conductance-cell module owning the programmed conductance buffer, the programming write, the per-call read snap, and the current and differential-conductance solve.

## Design decisions

- **`g_max__uS` is an `__init__` kwarg, not a config field.** The ceiling is a design value, not an intrinsic device parameter, so it stays outside the frozen `RramConfig` per the project rule that device design parameters are explicit constructor arguments. `g_min__uS` (intrinsic) lives in the config. The constructor enforces `g_max__uS > config.g_min__uS`.
- **Policy is separate from config.** `RramConfig` carries the physics; `RramPolicy` carries the per-run decision of which non-idealities are active, as flat `bool` fields with no defaults. The policy is loaded independently and is never embedded in the device physics.
- **Program-time vs read-time sources are partitioned by surface.** Programming Gamma, drift, and stuck-at are baked into `g__uS` by `program(...)`; telegraph and thermal are resampled per `snapshot(...)`. This split is a contract, not an accident: re-running `snapshot` must not re-roll the programmed-in faults.

## Contracts & invariants

- **`program(...)` mutates state by buffer reassignment.** The stored conductance is replaced (`self.g__uS = ...`), not edited in place, so the buffer can grow from its scalar-zero initial shape to the programmed shape. A view obtained before `program(...)` does not represent the replacement buffer.
- **`snapshot(shape, multi_coords)` is the only read path into fabricated state.** It expands `g__uS` to the per-call broadcast `shape`, optionally advanced-indexes a chunk via `multi_coords`, applies the read-time noise stack, and re-clamps. `multi_coords=None` returns the full broadcast view. The returned snapshot contains the read state; it does not register or duplicate device buffers.
- **`solve_dc(v, snap)` is stateless in the device.** It reads conductance only from the passed `RramSnap`, never from `self.g__uS`, so a chunk's snap and its solve stay paired.
- **No static fabricate mismatch.** `Rram` joins the fabricate cascade but declares an explicit no-op `_sample_fabricate_mismatch` — RRAM variation enters through `program(...)` (state-dependent Gamma, stuck-at), not through `fabricate()`.

## Performance & resources

The state is one conductance buffer at the programmed broadcast shape. The read snap allocates noise draws at the per-call `shape` (or the indexed chunk); the chunked-solve `multi_coords` path sizes that allocation per chunk rather than over the full leading batch.

## Gotchas

- **Drift is unconditional, not policy-gated.** Unlike the four policy-flagged sources, the power-law drift gain is applied whenever `drift_decay_rate > 0` and `t_elapsed > drift_t0` — there is no `RramPolicy` switch for it. To disable drift, set the device parameters, not a policy flag.
- **`alpha == 0` is a distinct branch.** The linear I-V path returns `g.expand_as(i)` for the differential conductance; do not assume the `sinh`/`cosh` form is always taken. The branch is on the config value, so it is compile-time-constant per instance.
- **Read noise is reseeded every snapshot.** Two snaps of the same programmed state differ under a noise-on policy; chunked reads are therefore not bit-identical to a single-block read with noise on.

## Known limitations

- No dedicated device-level test module exists. Current integration coverage exercises construction, programming, fabricated-state traversal, and cell-level use; direct I-V, write fixed-point, and per-source noise-statistics tests remain absent.

---

- **Reference**: [rram](../../../reference/primitive/device/rram.md)
- **Implementation**: `neurox/primitive/device/rram.py`
- **Tests**: `tests/primitive/xbar/test_array_fabricate.py`, `tests/primitive/xbar/test_cell_detail.py`, `tests/primitive/xbar/test_cell_linear.py`
