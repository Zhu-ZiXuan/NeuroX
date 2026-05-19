# State Holding

This document records the current state-ownership rules.

## Three stages of physical state

Every circuit / device's physical state moves through three explicit stages. Each stage has its own representation, its own write moment, and its own name form.

### 1. Nominal value (design intent)

The ideal value before any manufacturing variation. Set once and never overwritten.

- For a circuit parameter, this is the design value chosen from the config (a Python scalar).
- For a programmed device such as RRAM, this is the looked-up nominal conductance for each programmed state (a Tensor).

**Write moment**: a circuit's `__init__`; the first step of an RRAM `program(...)` call.

**Naming**: Python scalar `<name>__<unit>`. Tensor buffer `nominal_<name>__<unit>` (registered as a non-persistent buffer so `.to(device)` migrates it).

### 2. Actual value (post-fabrication / post-programming)

The nominal value plus static manufacturing variation (Pelgrom mismatch, write-noise, retention drift baked into the programmed state, etc.). This is the "true" value present in the silicon for the lifetime of the fabricated module.

**Write moment**: `fabricate(shape)` for circuits / sized devices; `program(...)` for RRAM-style write paths.

The actual value is sampled from the nominal value plus the configured static noise distribution. It is registered as a non-persistent buffer so it survives `.to(device)`. It is re-sampled cleanly on every `fabricate(...)` call (see [`fabrication_lifecycle.md`](fabrication_lifecycle.md)).

**Naming**: `<name>__<unit>` (no prefix). Same name as the nominal but without the `nominal_` prefix.

### 3. Snapshot value (per-call dynamic noise)

The actual value plus dynamic random noise (kT/C thermal noise, RTN / flicker noise, comparator decision noise, etc.) — the value the circuit "sees" at one specific moment during one DC solve.

**Write moment**: `snapshot(*, shape)`. Returned as a frozen `*Snapshot` dataclass and consumed by `solve_dc(...)` / `convert(...)` / similar primary methods.

**Naming**: fields of a frozen `*Snapshot` dataclass. Fields may carry tensors or nested `*Snapshot` instances; Python objects are not allowed.

The snapshot is **never** registered as a buffer. The solver iterates on a fixed snapshot — the circuit's electrical characteristic must not change between Newton iterations of the same call.

## Ownership rule

Fabricated state lives inside the module that physically owns it. Parents do not mirror child buffers.

Examples:

- `RRAM` owns programmed conductance tensors.
- `NMOS` owns fabricated `beta__uA_per_V2` / `vth__V`.
- `OpAmpTIA` owns its fabricated op-amp gain buffer.

A parent obtains child state through the child object, not through a duplicate buffer on itself.

## Snapshot pattern

`snapshot(*, shape: tuple[int, ...]) -> <Name>Snapshot` is the canonical signature.

- `shape` is the per-call broadcast shape for the snapshot tensors.
- The returned dataclass is `@dataclass(frozen=True)`.
- The fields are either `Tensor` or nested `*Snapshot`.
- The snapshot is consumed by `solve_dc(...)` / `convert(...)` etc. and discarded with the call. It is never persisted on the module.

## Cross-module snapshot passing

Some solvers iterate on snapshots that belong to different modules (e.g. `NewtonRaphsonSolver1T1R.solve_dc` reads `RRAMSnapshot`, `NMOSSnapshot`, plus two clamp-driver snapshots). Passing the snapshot across module boundaries is allowed; it is not a separate convention beyond the snapshot rules above. The snapshot type carries its own semantics and the receiver's signature documents which type it expects.

## Per-call runtime state

Per-call state — anything created inside `solve_dc(...)` / `convert(...)` / `vec_mat_mul(...)` other than the snapshot itself — is held in local variables, returned via the `*DCOP` / `*Output` / output Tensor, or emitted via the profiler side channel. It is never reassigned to module attributes.

## Why this split exists

- The owner is the only place that knows how to fabricate and interpret its own state.
- Parents orchestrate; they do not duplicate.
- Runtime solver code works against frozen snapshots — no aliasing, no surprise updates mid-iteration.
- A future functional "program/fabricate" + "run" split can reuse the same boundaries without redesigning every child.
