# Physical state

Physical modules distinguish immutable tensor sources from lifecycle-produced state. Nominal fabrication sources and fixed model tables are buffers so construction-time `dtype` and a pre-materialization `to(device)` determine where later work runs. Fabricated and programmed values are ordinary tensor attributes created by their lifecycle methods. Per-call snaps remain local values.

## State categories

- **Fabrication source** — an immutable `nominal_*` buffer used by `fabricate()`. A scalar physical baseline is normally 0-D; a genuinely multi-valued baseline, such as a reference ladder or capacitor bank, remains a compact vector or table.
- **Fixed model tensor** — an immutable buffer used directly by the model, such as a transfer LUT, state map, row mask, or wire-parameter vector. It is not part of the nominal/actual/snap progression.
- **Fabricated state** — an ordinary tensor attribute created or replaced by `fabricate()`. It contains static per-instance mismatch around a nominal source and normally has `inst_shape`.
- **Programmed state** — an ordinary tensor attribute created or replaced by `program(...)`. It contains the value written by the caller after mapping and programming effects.
- **Snap** — a fresh local tensor or frozen dataclass created for one call. It adds dynamic noise to fabricated or programmed state and is never stored on the module.
- **Learned runtime statistics** — observer state and similar PyTorch-managed statistics remain buffers because their lifecycle is training/calibration rather than physical fabrication or programming.

A leaf uses only the categories it needs. A programmable leaf may have no nominal source, while a deterministic LUT leaf may have no fabricated state.

## Lifecycle

The supported order is:

```text
construct -> to(device) -> fabricate -> program -> execute
```

`fabricate` or `program` may be absent when a model has no corresponding state.

### Construction

`__init__` binds config, policy, instance shape, immutable model metadata, child modules, and immutable buffers. It does not allocate instance-shaped placeholders for future fabricated or programmed state. Consequently, a stateful run path has no defined pre-fabrication or pre-program behavior.

Nominal sources are registered directly at their intended dtype. Code must not recover a source's dtype or device from an unrelated runtime tensor, and must not recreate a fixed source from Python data inside `fabricate`, `program`, or the execution path.

### Device migration

Call `to(device)` after construction and before materializing physical state. PyTorch migrates parameters and registered buffers, which moves every nominal source and fixed model tensor. Later lifecycle methods derive their tensors from those migrated sources or accept an already placed programming input.

Fabricated and programmed states are ordinary attributes, so a later `to(device)` does not migrate them. Code relying on migration after state materialization is unsupported. If migration is unavoidable, move the module and then rerun `fabricate()` and `program(...)` before execution.

### Fabrication

`fabricate()` resamples local static mismatch from unchanged nominal buffers and assigns the resulting tensors to fabricated-state attributes. The call traverses fabricable child modules in pre-order. Repeated calls replace prior realizations rather than perturbing them cumulatively.

### Programming

`program(...)` creates or replaces programmed-state attributes. Programming inputs must already be on the intended device and use the intended dtype unless the public method explicitly defines a value-domain conversion. A module must not use an unrelated tensor as an implicit placement anchor.

Programming is dispatched by the owner rather than cascaded uniformly because each owner organizes a different logical value for its children. Repeated calls replace prior programmed state.

### Snapshot

`snapshot(...)` derives one call-local view from fabricated or programmed state, optionally expands it to the call shape, and samples dynamic noise. A snap is not cached, registered, or persisted.

## Persistence and ownership

Nominal and fixed model buffers are normally `persistent=False`; they are reproducible from config and construction arguments. Fabricated and programmed ordinary attributes are absent from `state_dict` by construction. The persisted upper-layer weight remains the source of truth, and loading a checkpoint must be followed by the normal fabrication/programming lifecycle.

Physical state lives only on its owning module. A parent delegates to a child or consumes the child's public snap; it does not mirror the child's state.

## Parallel execution

Each data-parallel rank materializes its own ordinary fabricated/programmed state. Framework buffer broadcast does not synchronize those attributes. Coordinate seeds and lifecycle calls explicitly when ranks must share one physical realization.
