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

## Persistence and ownership

Nominal, functional, and circuit constant buffers are normally `persistent=False`; they are reproducible from config and construction arguments. Fabricated and programmed ordinary attributes are absent from `state_dict` by construction. The persisted upper-layer weight remains the source of truth, and loading a checkpoint must be followed by the normal fabrication/programming lifecycle.

Physical state lives only on its owning module. A parent delegates to a child or consumes the child's public snap; it does not mirror the child's state.

## Parallel execution

Each data-parallel rank materializes its own ordinary fabricated/programmed state. Framework buffer broadcast does not synchronize those attributes. Coordinate seeds and lifecycle calls explicitly when ranks must share one physical realization.
