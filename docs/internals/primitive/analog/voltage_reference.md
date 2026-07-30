# Voltage reference

`Vref` is a leaf multi-output source holding a 2-D `[mode][tap]` bank and exposing one selected mode through `VrefSnap`. It is structurally identical to `Iref`; only the dimension of the sourced quantity differs.

## Design decisions

- **PPA is static-only; nothing to log per call.** The reference emits no dynamic energy and no latency, so its area and always-on bias power ride the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. There is no `v_supply__V` field and no bias-current field.
- **Two-stage sampling split.** The initial-accuracy spread is sampled once in `_sample_fabricate_mismatch` and held as `_v_refs__V` fabricated state; the per-call noise is resampled every `snapshot`. This fabricate-then-runtime split holds a die's tolerance fixed across reads while the per-call noise varies per call.
- **Both departures are relative (multiplicative).** `apply_relative_gaussian` scales by `1 + randn * sigma` at each stage, so one sigma applies uniformly to taps of differing magnitude and a `0` V tap stays exactly zero; an absolute per-tap sigma would need a tuple.
- **Static mismatch spans `inst_shape`; per-call noise spans the full call shape.** The two draws describe different physical quantities and are simultaneously correct: the fabricated taps are one physical generator per instance, while every position of a call outside `inst_shape` is a distinct instant (time-serial on that hardware) or a distinct point of the distribution net, and thermal fluctuation is not reused across instants. The corollary keeps the books straight — a *static* per-reader offset belongs to the reader, which already fabricates its own offset over its own `inst_shape`, so the source deliberately carries no per-consumer static axis; adding one would double-count the same deviation.
- **`snapshot(*, mode, shape)` selects the taps inside the source.** The caller names the mode, the source resolves it, so the bank layout stays private and no operating-mode identity travels downstream with the taps. `shape` is mandatory and is the full output shape.
- **One instance serves one consumer.** A reference source is fabricated for the circuit it feeds, so it never acts as a shared multi-consumer bank; a second consumer gets its own instance with its own static seat.

## Contracts & invariants

- **State lifecycle.** `_nominal_v_refs__V` is a non-persistent nominal buffer with shape `(mode_num, tap_num)`. `fabricate()` clone-expands it to `(*inst_shape, mode_num, tap_num)` and assigns `_v_refs__V` ordinary state, with tolerance applied when enabled. The clone keeps an in-place write on the fabricated state — or on a snap that aliases it — from reaching the registered buffer. Snapshotting requires fabrication first.
- **`mode_num` / `tap_num` are properties** on config and module (`len(config.v_refs__V)` / `len(config.v_refs__V[0])`) and define the bank geometry.
- **Read path is `snapshot(mode=..., shape=...).v_refs__V`.** The returned tensor is exactly `shape`, whose last axis is `tap_num`; the mode axis is already resolved away, and the `inst_shape` prefix must right-align inside `shape`. Range-checking `mode` belongs to the consumer that owns the mode set.
- **A clamp consumer supplies the driver's call shape.** A single-tap bank feeding a boundary clamp is snapshotted at `(..., 1)` and the trailing tap axis is consumed away, because the clamp driver takes no `shape` of its own — the shape it samples at is the one its injected reference carries.
- **No `multi_coords`, unlike device and cell snapshots.** `multi_coords` exists to align a snap with one solver *chunk*, and reference sources sit outside the solver's chunk loop. Two classes result: inside the loop (cell grids, orders of magnitude larger) a snapshot takes `shape` plus `multi_coords`; outside it (reference sources, clamp drivers) the shape is carried by the injected tensor alone. Reference and clamp tensors are not chunked because they are orders of magnitude smaller than the cell grid.
- **Modes are equal-length and non-negative**, validated at config time (`>= 1` mode, `>= 1` tap per mode, equal tap lengths, each tap `>= 0`). Ordering within a mode is deliberately **not** validated here: what a mode means is the consumer's knowledge, so a decision ladder's ascent is checked at config time by the converter or macro that wires it. The field is always an explicit 2-D bank, including for the degenerate single-mode single-tap `[[v]]`; a flat TOML array is rejected by deserialization.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference has no `_v_refs__V` state; call `fabricate()` before snapshotting.
- **With both stages off the snap is a view, not a copy.** `apply_relative_gaussian` returns its input untouched when disabled, so the snap can alias the expanded fabricated state (itself an expanded view of a private clone of the nominal buffer when `tolerance` is off). That is what keeps a wide call shape free of a materialized allocation; never mutate a snap tensor in place.

---

- **Reference**: [voltage_reference](../../../reference/primitive/analog/voltage_reference.md)
- **Implementation**: `neurox/primitive/analog/voltage_reference.py`
- **Tests**: `tests/primitive/analog/test_voltage_reference.py`
