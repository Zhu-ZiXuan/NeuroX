# Current reference

`Iref` is a leaf multi-output source holding a 2-D `[mode][tap]` bank and exposing one selected mode through `IrefSnap`.

## Design decisions

- **PPA is static-only; no dynamic energy, no latency.** Area and the always-on bias power live in the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. The bias power that generates the reference currents is folded into `leakage_per_inst__uW` and is **not** derived from the tap values. There is no `v_supply__V` field. The conduction energy of a reference current flowing through a reading circuit is billed by that circuit.
- **Two-stage non-ideality split mirrors the fabrication lifecycle.** The initial-accuracy spread is a per-die constant sampled once in `_sample_fabricate_mismatch` and stored as ordinary fabricated state; the per-call noise is resampled every `snapshot`.
- **One relative (multiplicative) sigma per stage.** A single scalar covers taps of differing magnitude; an absolute departure would instead need a per-tap tuple. `apply_relative_gaussian` also keeps a `0` uA tap exact.
- **Static mismatch spans `inst_shape`; per-call noise spans the full call shape.** The two draws describe different physical quantities and are simultaneously correct: the fabricated taps are one physical generator per instance, while every position of a call outside `inst_shape` is a distinct instant (time-serial on that hardware) or a distinct mirrored branch, and thermal fluctuation is not reused across instants. The corollary keeps the books straight — a *static* per-reader offset belongs to the reader, whose own `inst_shape` already carries a per-IO axis and its comparator / coupling offsets, so the source deliberately carries no per-consumer static axis; adding one would double-count the same deviation.
- **`snapshot(*, mode, shape)` selects the taps inside the source.** The caller names the mode, the source resolves it, so the bank layout stays private and no operating-mode identity travels downstream with the taps. `shape` is mandatory and is the full output shape.
- **One instance serves one consumer.** A reference source is fabricated for the circuit it feeds, so it never acts as a shared multi-consumer bank; a second consumer gets its own instance with its own static seat.

## Contracts & invariants

- **State lifecycle.** `_nominal_i_refs__uA` is a non-persistent nominal buffer with shape `(mode_num, tap_num)`. `fabricate()` clone-expands it to `(*inst_shape, mode_num, tap_num)` and assigns `_i_refs__uA` ordinary state, with tolerance applied when enabled. The clone keeps an in-place write on the fabricated state — or on a snap that aliases it — from reaching the registered buffer. Snapshotting requires fabrication first.
- **`mode_num` / `tap_num` are properties** on config and module (`len(config.i_refs__uA)` / `len(config.i_refs__uA[0])`) and define the bank geometry.
- **Read path is `snapshot(mode=..., shape=...).i_refs__uA`.** The returned tensor is exactly `shape`, whose last axis is `tap_num`; the mode axis is already resolved away, and the `inst_shape` prefix must right-align inside `shape`. Range-checking `mode` belongs to the consumer that owns the mode set.
- **No `multi_coords`, unlike device and cell snapshots.** `multi_coords` exists to align a snap with one solver *chunk*, and reference sources sit outside the solver's chunk loop. Two classes result: inside the loop (cell grids, orders of magnitude larger) a snapshot takes `shape` plus `multi_coords`; outside it (reference sources, clamp drivers) the shape is carried by the injected tensor alone. Reference and clamp tensors are not chunked because they are orders of magnitude smaller than the cell grid.
- **Modes are equal-length and non-negative**, validated at config time (`>= 1` mode, `>= 1` tap per mode, equal tap lengths, each tap `>= 0`). Ordering within a mode is deliberately **not** validated here: what a mode means is the consumer's knowledge, so a decision ladder's ascent is checked at config time by the converter or macro that wires it. The field is always an explicit 2-D bank, including for a single mode; a flat TOML array is rejected by deserialization.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference has no `_i_refs__uA` state; call `fabricate()` before snapshotting.
- **With both stages off the snap is a view, not a copy.** `apply_relative_gaussian` returns its input untouched when disabled, so the snap can alias the expanded fabricated state (itself an expanded view of a private clone of the nominal buffer when `tolerance` is off). That is what keeps a wide call shape free of a materialized allocation; never mutate a snap tensor in place.

---

- **Reference**: [current_reference](../../../reference/primitive/analog/current_reference.md)
- **Implementation**: `neurox/primitive/analog/current_reference.py`
- **Tests**: `tests/primitive/analog/test_current_reference.py`
