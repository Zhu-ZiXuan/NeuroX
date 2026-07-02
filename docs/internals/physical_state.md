# Physical state

Physical-modeling modules carry their physical state in named buffers, and the lifecycle methods evolve each quantity from its ideal design value to the noisy value a run sees. A single physical quantity is held at three fidelity levels along one line: the ideal **nominal** value; the **actual** realization that either bakes in static per-instance mismatch or writes a programmed value; and the per-call **snap** that adds dynamic noise. Only physical quantities travel this line — integer weights, converter lookup tables, and config-derived constant tables are documented with the module that owns each, and a purely arithmetic module carries none of this machinery.

## State tiers

Buffers live on concrete leaves, not on the abstract bases above them. A leaf carries three tiers of one physical quantity:

- **Nominal** — the design value before any per-instance spread, held in a 0-d `nominal_*` template derived from config at construction. It never changes afterward and is the fixed reference the later tiers derive from.
- **Actual** — the realized electrical characteristic, held in a buffer that drops the `nominal_` prefix. A leaf takes exactly one of two branches: a fabricate buffer, into which fabrication samples static per-instance mismatch around the nominal template; or a program buffer, into which programming writes the target value. A leaf never carries both.
- **Snap** — the per-call value the run path sees, derived from the actual buffer at one call and carrying the dynamic noise present at that instant. It is produced fresh each call and never stored.

Freezing the actual buffer across a call lets the call's many iterations run against a fixed characteristic rather than one drifting between them.

Some programmable devices carry no nominal template and no fabricate buffer at all — only a program buffer initialized to zeros that receives its target value directly.

## Lifecycle methods

Four methods drive a module's state. `__init__` runs once at construction; `fabricate()` and `program(...)` run after it, in either order and any number of times; `snapshot(...)` runs once per call on the run path. `fabricate()` and `program(...)` write orthogonal state — mismatch versus programmed value — so their order is free.

### `__init__`

`__init__` binds config, commits the module's shape, and registers the buffers; an owning module also builds its children here. It registers the 0-d nominal template and, beside it, the actual placeholder: the fabricate branch initializes the placeholder to `nominal.clone()`, the program branch to zeros. Because the placeholder is well-defined from construction, a leaf produces sensible output before either `fabricate()` or `program(...)` has run — the mismatch-free nominal value on the fabricate branch, zero on the program branch. Each module family documents its own construction signature — the exact parameters and the child shapes a composite derives — in that family's own documentation.

### `fabricate()`

`fabricate()` resamples static per-instance mismatch from the unchanged nominal template into the fabricate buffer, and only fabricate-branch leaves carry it. One call cascades over the whole module tree, so a newly layered module participates without extra code.

### `program(...)`

`program(...)` writes the programmed value into the program buffer of a programmable leaf. Unlike `fabricate()` it is dispatched per layer rather than cascaded, because it consumes one logical weight that only the owning module can organize and encode; there is no tree-uniform argument to hand every child. The run path then reads this established value rather than receiving the weight as an argument.

### `snapshot(...)`

`snapshot(*, shape)` derives the per-call snap from the actual buffer. `fabricate()` samples static mismatch at the construction-time `inst_shape`; `snapshot` then expands and samples the dynamic noise over the per-call (serial-execution) broadcast `shape` on top of that realization. A snap's fields are tensors or nested snaps only, so the whole structure is device-migratable. A snap is valid for exactly one call — its memory scales with the call's leading batch rather than `inst_shape` and is freed when the call returns, and caching one across calls is a bug. A snap may cross module boundaries, read by a module other than the one that owns it, each snap type carrying its own semantics and documented by the receiver's signature. The concrete fields of each snap belong to the leaf that produces it.

## Buffer reassignment and idempotency

A module holds two groups of buffers — the nominal templates and their actual counterparts — and the two counts need not match: a module may template several quantities, and a program buffer has no template at all, so the nominal-to-actual mapping is per module, not one-to-one. Both groups are registered `persistent=False`.

The actual buffer is updated by attribute reassignment — `self.x = new_tensor` — never by a second `register_buffer`.

`fabricate()` and `program(...)` may each run any number of times, and every call resamples or re-encodes from the unchanged nominal template, reassigning the actual buffer without accumulating state — repeated fabrication and programming are idempotent in distribution, a fresh draw rather than an additive update.

Because the actual buffers are `persistent=False`, they stay out of `state_dict`. The upper-layer weight is the persisted source of truth; a `program(...)` at checkpoint-load time regenerates the per-cell state, so checkpoints stay independent of device and geometry.

## State ownership

Physical state lives in the owning module; a parent reads a child's state through the child, never mirroring it.

## Call cadence and determinism

Resampling stays off the inference hot path — `program(...)` and `fabricate()` run before inference, neither during it. Each `fabricate()` is a single pass over the module tree whose allocation scales with `inst_shape`, not with any per-call batch.

Fabrication draws its mismatch from nondeterministic RNG, so it conflicts with `torch.use_deterministic_algorithms(True)` — disable deterministic mode while fabricating. Across data-parallel ranks each rank draws its own mismatch realization and the lifecycle does not synchronize, so coordinate RNG seeds before fabricating if cross-rank-consistent mismatch is required.
