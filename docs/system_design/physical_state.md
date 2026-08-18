# Physical state

A physical module holds two kinds of tensor state, and the split decides where a value lives, when it is sampled, and on which device it runs.

Registered buffers carry what construction fixes: nominal baselines, functional tables, and circuit constants. Each is registered at its intended `dtype`, and PyTorch migrates it with the module, so a `to(device)` taken before any physical state exists decides where all later work runs. Fabricated and programmed values are ordinary tensor attributes written by the lifecycle method that produces them, and a per-call snap is a local value no module keeps.

All of it is implementation state of the owning module. A consumer reads it through that module's `*Snap` or `*Dcop` value object, or through a purpose-built accessor when the value is part of the module's contract. Concrete cross-module value objects inherit `SnapBase` or `DcopBase`; a `Protocol` that only describes the structural surface a collaborator accepts does not. Registered child modules are the exception: they keep public structural names, because PyTorch addresses them by name in the module tree.

## State categories

- **Functional buffer** — read directly by the forward math: a transfer LUT, a state map, an index or mapping mask, a bias constant.
- **Circuit constant buffer** — electrical and timing constants, such as a wire-parameter profile or a frozen output resistance.
- **Nominal buffer** — the `_nominal_*` baseline `fabricate()` consumes. A scalar physical baseline is 0-D; a genuinely multi-valued one, such as a reference ladder or a capacitor bank, stays a compact vector or table.
- **Fabricated state** — the attribute `fabricate()` writes: static per-instance mismatch around a nominal buffer, carrying the instance axes.
- **Programmed state** — the attribute `program(...)` writes: the caller's value after mapping and programming effects.
- **Snap** — a fresh tensor or frozen value object built for one call, adding dynamic noise on top of fabricated or programmed state, never stored on the module.

Functional and circuit constant buffers stand outside the nominal-to-fabricated-to-snap progression, and a module declares only the categories its model needs: a programmable leaf may have no nominal buffer, and a deterministic LUT leaf no fabricated state. A class header groups its tensor declarations under these same category names, in the banner form fixed by [code_style](../conventions/code_style.md).

## Lifecycle

```text
construct -> to(device) -> fabricate -> program -> execute
```

### Construction

`__init__` binds config, policy, instance shape, compact scalar metadata, child modules, and registered buffers. It allocates no placeholder for state a later stage produces, so a run that skips a stage raises on the missing attribute instead of computing on a stand-in.

Placement and precision are settled once, at the module's own buffers. A module must not recover a `dtype` or a device from an unrelated runtime tensor, and must not rebuild a fixed buffer from Python data inside `fabricate`, `program`, or the execution path.

### Device migration

`to(device)` moves parameters and registered buffers, which covers every nominal, functional, and circuit constant buffer; later lifecycle methods then derive their tensors from the migrated buffers or accept an already placed programming input. Fabricated and programmed states are ordinary attributes, so a `to(device)` after they exist leaves them behind. Migration after materialization is unsupported: move the module first, then rerun `fabricate()` and `program(...)`.

### Fabrication and programming

`neurox.fabricate(model)` visits every NeuroX module registered anywhere in a user model; `module.fabricate()` applies the same operation to one NeuroX subtree. Both resample from unchanged nominal buffers in registered-module pre-order, which is why a repeated call replaces the previous realization instead of compounding onto it. A module with no local fabricated state inherits the empty sampling hook while its descendants still participate. Skipping fabrication leaves produced attributes absent and fails at their first use; zero manufacturing variation is selected by policy and still passes through this lifecycle stage.

`program(...)` does not cascade, because each owner organizes a different logical value for its children — a weight grid becomes per-cell state indices, an index becomes a target conductance — and only the owner knows that mapping. Each owner therefore dispatches to its children explicitly. Programming input must already carry the intended device and dtype unless the receiving method defines a value-domain conversion, and a repeated call replaces the previous programmed state.

## Who samples, and when

A random draw is taken once per physical instance at fabricate time, or once per access at snapshot time. Nothing between those two events may re-roll either, and the difference in what each expands over is what keeps a static deviation static:

- `fabricate()` expands over **space** — the instance axes alone. One draw per physical instance, held until the next call.
- `snapshot(...)` expands over **space and time** — the instance axes plus every access position the call covers. One fresh dynamic draw per access position, so the count of time positions is the count of accesses.

Circuit time and space axes are orthogonal to the serial and parallel axes of programming and of chunking. A chunk axis is a memory tiling of positions that already exist, so it neither adds an access nor takes a draw. One `snapshot(...)` call draws once, at the full access shape it is handed, ahead of any tiling a consumer applies to the result; the chunk size therefore never changes how many sampling events a run takes.

### The owner of the event structure samples

How many accesses one call covers, and which of them are correlated, lives with the module scheduling the accesses — a composing macro — and nowhere below it. A macro therefore expands its references to the event shape, snapshots the blocks it owns, folds the inserted axis away if there is one, and passes finished snaps down.

A module that receives a snap performs no boundary-shape normalization in return. Widening a narrow reference on arrival would silently assert a correlated reading — one draw shared across positions that are physically distinct — on behalf of every caller that had not thought about it, which is precisely the decision the caller is supposed to make explicitly.

### Sources and drivers

Static shared identity belongs to the source, and dynamic per-access noise belongs to the consuming driver. A reference source is consequently fabricate-only: one bank per physical instance, sampled once in its fabrication hook, fanned out to its consumers by views, exposed through a read-only accessor rather than a `snapshot`, and never resampled. A driver is what knows what an access is, so a driver is where the per-access draw happens.

The split also fixes which axes each end carries. A source spans its instance axes and nothing further — no per-consumer axis — because a static deviation that differs between the circuits reading one source is the reading circuit's own fabricated state, over its own instance axes. Carrying it at both ends would count one physical deviation twice.

### Snap shapes

A snap's shape is `inst_shape`-compatible: axes are added around `inst_shape`, never resized. A time axis sits in the leading don't-care region and its positions are right-anchored, shape growth being prepend-only; a time axis that will later merge with axes to its right sits immediately left of them.

An inserted time axis lives only between the snapshot statement and the fold statement that merges it away. Every function boundary therefore carries the canonical `[..., *trailing]` shape, and no signature anywhere gains an argument for an axis that exists across two statements.

Right-anchoring is stated for a fold target in the trailing region. Where the axes a time axis merges with are a suffix of the module's own `inst_shape`, "immediately left of the axes it merges with" places the time axis inside that instance prefix, which `inst_shape` compatibility does not admit; a module carrying both a fabrication prefix and a live static draw would mis-seat the axis or raise.

TODO (mismatch modeling): resolve the placement for a suffix fold target once noise and mismatch modeling is frozen.

## Persistence and ownership

No physical state is persisted. Buffers are registered non-persistent because config and construction arguments reproduce them exactly, and fabricated and programmed attributes stay outside `state_dict` by construction. Restoring a model therefore restores the caller's own weights, which remain the source of truth, and the fabricate and program lifecycle runs again afterwards. For the same reason, framework machinery that broadcasts or synchronizes parameters and buffers does not reach fabricated or programmed state.

Physical state lives only on its owning module. A parent delegates to a child or consumes the child's public snap; it does not mirror the child's state.
