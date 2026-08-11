# Physical state

Physical modules distinguish registered buffers from lifecycle-produced state. Nominal, functional, and circuit constant tensors are buffers so construction-time `dtype` and a pre-materialization `to(device)` determine where later work runs. Fabricated and programmed values are ordinary tensor attributes created by their lifecycle methods. Per-call snaps remain local values.

Module-owned nominal, functional, circuit constant, fabricated, and programmed
tensors are implementation state. Consumers read them through the relevant
`*Snap` or `*Dcop` value object, or through a purpose-specific method/property
when the value is part of the module contract. Registered child `nn.Module`
attributes remain structural names because PyTorch uses them in the module tree.

## State categories

- **Functional buffer** — a registered buffer the forward math reads directly, such as a transfer LUT, state map, index or mapping mask, or bias constant.
- **Circuit constant buffer** — a registered buffer holding electrical and timing constants, such as a wire-parameter profile or a frozen output resistance.
- **Nominal buffer** — a registered `nominal_*` buffer `fabricate()` consumes. A scalar physical baseline is normally 0-D; a genuinely multi-valued baseline, such as a reference ladder or capacitor bank, remains a compact vector or table.
- **Fabricated state** — an ordinary tensor attribute created or replaced by `fabricate()`. It contains static per-instance mismatch around a nominal buffer and normally has `inst_shape`.
- **Programmed state** — an ordinary tensor attribute created or replaced by `program(...)`. It contains the value written by the caller after mapping and programming effects.
- **Runtime buffer** — a registered buffer the forward mutates in place. Observer state and similar PyTorch-managed statistics are buffers because their lifecycle is training or calibration rather than physical fabrication or programming.
- **Snap** — a fresh local tensor or frozen dataclass created for one call. It adds dynamic noise to fabricated or programmed state and is never stored on the module.

Functional, circuit constant, and runtime buffers stand outside the nominal/actual/snap progression. A leaf uses only the categories it needs. A programmable leaf may have no nominal buffer, while a deterministic LUT leaf may have no fabricated state.

## Lifecycle

The supported order is:

```text
construct -> to(device) -> fabricate -> program -> execute
```

`fabricate` or `program` may be absent when a model has no corresponding state.

### Construction

`__init__` binds config, policy, instance shape, compact scalar metadata, child modules, and registered buffers. It does not allocate instance-shaped placeholders for future fabricated or programmed state. Consequently, a stateful run path has no defined pre-fabrication or pre-program behavior.

Nominal buffers are registered directly at their intended dtype. Code must not recover a buffer's dtype or device from an unrelated runtime tensor, and must not recreate a fixed buffer from Python data inside `fabricate`, `program`, or the execution path.

### Device migration

Call `to(device)` after construction and before materializing physical state. PyTorch migrates parameters and registered buffers, which moves every nominal, functional, and circuit constant buffer. Later lifecycle methods derive their tensors from those migrated buffers or accept an already placed programming input.

Fabricated and programmed states are ordinary attributes, so a later `to(device)` does not migrate them. Code relying on migration after state materialization is unsupported. If migration is unavoidable, move the module and then rerun `fabricate()` and `program(...)` before execution.

### Fabrication

`fabricate()` resamples local static mismatch from unchanged nominal buffers and assigns the resulting tensors to fabricated-state attributes. The call traverses fabricable child modules in pre-order. Repeated calls replace prior realizations rather than perturbing them cumulatively.

### Programming

`program(...)` creates or replaces programmed-state attributes. Programming inputs must already be on the intended device and use the intended dtype unless the public method explicitly defines a value-domain conversion. A module must not use an unrelated tensor as an implicit placement anchor.

Programming is dispatched by the owner rather than cascaded uniformly because each owner organizes a different logical value for its children. Repeated calls replace prior programmed state.

### Snapshot

`snapshot(...)` derives one call-local view from fabricated or programmed state, optionally expands it to the call shape, and samples dynamic noise. A snap is not cached, registered, or persisted.

## Sampling axes

Fabrication and snapshotting expand over different axis sets, and that difference is what keeps a static deviation static:

- `fabricate()` expands over **space** — the instance axes alone. One draw per physical instance, held until the next call.
- `snapshot(...)` expands over **space and time** — the instance axes plus every access position the call covers. One fresh dynamic draw per access position, so the count of time positions is the count of accesses.

Circuit time and space axes are orthogonal to the serial and parallel axes of programming and of chunking. A chunk axis is a memory tiling of positions that already exist, so it neither adds an access nor takes a draw.

### Snap shapes

A snap's shape is `inst_shape`-compatible: axes are added around `inst_shape`, never resized. Time axes sit in the leading don't-care region and their positions are right-anchored, shape growth being prepend-only; a time axis that will later merge with axes to its right sits immediately left of them.

An inserted time axis lives only between the snapshot statement and the fold statement that merges it away. Every function boundary therefore carries the canonical `[..., *trailing]` shape, and no signature anywhere gains an argument for an axis that exists across two statements.

### Who takes the snapshot

The owner of the event structure does. How many accesses one call covers, and which axes are correlated across them, is knowledge that lives with the module that schedules the accesses — a composing macro — and nowhere below it. So a macro expands its references to the event shape, snapshots the blocks it owns, folds the inserted axis if there is one, and passes finished snaps down.

A consumer that receives a snap therefore performs no boundary-shape normalization. Widening a narrow reference on arrival would silently assert the correlated reading — one draw shared across positions that are physically distinct — for every caller that had not thought about it, which is precisely the decision the caller is supposed to make explicitly.

### Sources versus drivers

Static shared identity belongs to the source, and dynamic per-access noise belongs to the consuming driver. A reference source is consequently fabricate-only: one bank per physical instance, sampled once, fanned out to its consumers by views and never resampled, exposed through a read-only accessor rather than a `snapshot`. A driver is what knows what an access is, so a driver is where the per-access draw happens.

### Known limitation

The right-anchoring rule is stated for a fold target in the trailing region. Where the axes a time axis will merge with are a suffix of the module's own `inst_shape`, "immediately left of the axes it merges with" places the time axis inside that prefix, which is not a position the `inst_shape`-compatibility rule admits. Under structurally zero static state — a nominal 0-dim buffer with its policy switch off — both placements give identical numbers, so the arrangement is inert; a module carrying both a fabrication prefix and a live static draw would mis-seat or raise.

TODO (mismatch modeling): resolve the placement for a suffix fold target once noise and mismatch modeling is frozen.

## Persistence and ownership

Nominal, functional, and circuit constant buffers are normally `persistent=False`; they are reproducible from config and construction arguments. Fabricated and programmed ordinary attributes are absent from `state_dict` by construction. The persisted upper-layer weight remains the source of truth, and loading a checkpoint must be followed by the normal fabrication/programming lifecycle.

Physical state lives only on its owning module. A parent delegates to a child or consumes the child's public snap; it does not mirror the child's state.

## Parallel execution

Each data-parallel rank materializes its own ordinary fabricated/programmed state. Framework buffer broadcast does not synchronize those attributes. Coordinate seeds and lifecycle calls explicitly when ranks must share one physical realization.
