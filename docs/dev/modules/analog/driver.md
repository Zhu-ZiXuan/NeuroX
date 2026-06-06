# `neurox/analog/driver.py`

## Current role

`Driver` is the ideal constant-voltage clamp driver. It implements the [`ClampDriver`](clamp_driver.md) protocol with no feedback loop and no runtime state beyond a single sampled clamp voltage.

## What it owns

- `config: DriverConfig` — drive voltage, thermal-noise sigma, and PPA / spec fields.
- `policy: DriverPolicy` — single `drive_thermal: bool` field, gates the per-snapshot Gaussian.
- `nominal_drive_value` — a 0-d non-persistent buffer at `config.drive_value`. Fabricated state is just this scalar; the driver has no shape-dependent state.
- per-call `DriverSnapshot` carrying the sampled clamp voltage.

## Clamp-driver protocol

The driver satisfies the family-wide `ClampDriver` Protocol surface:

- `v_ref__V` — ideal / zero-current clamp voltage (returns `config.drive_value`).
- `fabricate()` — inherited auto-cascade from `FabricateMixin`; the ideal driver has no static state, so its `_sample_fabricate_mismatch` is the default no-op.
- `snapshot(*, shape)` — samples a `DriverSnapshot`; applies `config.drive_thermal` if configured.
- `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V=None)` — returns `(v_clamp__V, dVclamp_dI__MOhm)` where the second tensor is always zero (the ideal driver is a voltage source).
- `solve_dc(...)` — richer dataclass entry point available on the concrete handle.

## Construction

`Driver.__init__(*, config, policy, name, inst_shape, dtype, T__K)` — `inst_shape` is committed here; the driver has no shape-dependent fabricated buffer.

## Why this is not a TIA

The clamp-driver protocol is intentionally separate from the TIA family. Both are clamp-driver protocol implementations, but a TIA closes a feedback loop on a virtual-ground reference while a `Driver` is an ideal voltage source — different physics, different parameter spaces. They share only the `ClampDriver` solver-facing contract.

See also:

- `clamp_driver.md`
- `docs/dev/architecture/config_and_construction.md`
