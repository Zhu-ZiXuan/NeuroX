# `neurox/analog/driver.py`

## Current role

`Driver` is the ideal constant-voltage clamp driver — no feedback loop, no runtime state beyond a single sampled clamp voltage. The 1T1R solver binds it as the SL boundary actor.

## What it owns

- `config: DriverConfig` — drive voltage, thermal-noise sigma, and PPA / spec fields.
- `policy: DriverPolicy` — single `drive_thermal: bool` field, gates the per-snapshot Gaussian.
- `nominal_drive_value` — a 0-d non-persistent buffer at `config.drive_value`. Fabricated state is just this scalar; the driver has no shape-dependent state.
- per-call `DriverSnapshot` carrying the sampled clamp voltage.

## Solver-facing surface

- `v_ref__V` — ideal / zero-current clamp voltage (returns `config.drive_value`).
- `fabricate()` — inherited auto-cascade from `FabricateMixin`; the ideal driver has no static state, so its `_sample_fabricate_mismatch` is the default no-op.
- `snapshot(*, shape)` — samples a `DriverSnapshot`; applies `config.drive_thermal` if configured.
- `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V=None)` — returns `(v_clamp__V, dVclamp_dI__MOhm)` where the second tensor is always zero (the ideal driver is a voltage source).
- `solve_dc(...)` — richer dataclass entry point available on the concrete handle.

## Construction

`Driver.__init__(*, config, policy, name, inst_shape, dtype, T__K)` — `inst_shape` is committed here; the driver has no shape-dependent fabricated buffer.

## Why this is not a TIA

`Driver` and `TIA` are physically different boundary actors: a TIA closes a feedback loop on a virtual-ground reference, while a `Driver` is an ideal voltage source — different physics, different parameter spaces. They share only the shape of the solver-facing `solve_clamp(...)` entry. The 1T1R solver binds `bl_driver: TIA` and `sl_driver: Driver` as concrete types — there is no shared abstract base because nothing else needs one.

See also:

- `docs/dev/architecture/config_and_construction.md`
