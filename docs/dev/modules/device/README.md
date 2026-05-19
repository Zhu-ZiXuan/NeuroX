# Device Models

This directory documents the current device-layer design.

Device modules own local physics and fabricated device state. They do not own mapping, array geometry, or circuit-specific orchestration.

Current rules:

- device configs contain process + spec parameters only
- device design parameters are explicit `__init__` arguments
- devices do **not** own a profiler `name`; their PPA rolls up to the consuming circuit
- fabricated state stays inside the owning device instance
- runtime snapshots are explicit per-call objects

Current files:

- [`nmos.md`](nmos.md) — NMOS transistor primitive (pure electrical).
- [`rram.md`](rram.md) — resistive memory cell with sinh I-V and the full noise stack.
- [`selector.md`](selector.md) — OTS-style threshold selector.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/architecture/state_holding.md`
- `docs/dev/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md`
