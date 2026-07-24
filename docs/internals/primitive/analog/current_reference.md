# Current reference

`CurrentReference` is a leaf multi-output source holding a 2-D `[mode][tap]` bank and exposing per-call tap values through `CurrentReferenceSnap`.

## Design decisions

- **PPA is static-only; no dynamic energy, no latency.** Area and the always-on bias power live in the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. The bias power that generates the reference currents is folded into `leakage_per_inst__uW` and is **not** derived from the tap values. There is no `v_supply__V` field.
- **Two-stage non-ideality split mirrors the fabrication lifecycle.** The initial-accuracy spread is a per-die constant sampled once in `_sample_fabricate_mismatch` and stored as ordinary fabricated state; the per-read noise is resampled every `snapshot`.
- **One relative (multiplicative) sigma per stage.** A single scalar covers taps of differing magnitude; an absolute departure would instead need a per-tap tuple.
- **`snapshot(shape=...)` expands, then draws noise.** The fabricated per-instance taps are expanded to the requested `shape` (intrinsic trailing `(mode_num, tap_num)`) before the per-call noise draw, so each position samples independently. An empty `shape` leaves the taps at the fabricated `(*inst_shape, mode_num, tap_num)`.

## Contracts & invariants

- **State lifecycle.** `nominal_i_refs__uA` is a non-persistent source buffer with shape `(mode_num, tap_num)`. `fabricate()` expands it to `(*inst_shape, mode_num, tap_num)` and assigns `i_refs__uA` as ordinary fabricated state, with tolerance applied when enabled. Snapshotting requires fabrication first.
- **`mode_num` / `tap_num` are properties** on config and module (`len(config.i_refs__uA)` / `len(config.i_refs__uA[0])`) and define the bank geometry.
- **Read path is `i_ref__uA(snap)`.** The accessor returns the full `(*inst_shape, mode_num, tap_num)` tensor from the snap rather than the buffer, so the per-call noise is always included.
- **Rows are strictly increasing, equal-length, and non-negative**, validated at config time (`>= 1` mode row, equal row lengths, per-row strictly increasing, each tap `>= 0`); a `0` uA first tap is permitted. The field is always an explicit 2-D bank, including for a single mode; a flat TOML array is rejected by deserialization.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference has no `i_refs__uA` state; call `fabricate()` before snapshotting.
- **`noise` off still clones.** A returned tensor never aliases the stored fabricated state.

---

- **Reference**: [current_reference](../../../reference/primitive/analog/current_reference.md)
- **Implementation**: `neurox/primitive/analog/current_reference.py`
- **Tests**: `tests/primitive/analog/test_current_reference.py`
