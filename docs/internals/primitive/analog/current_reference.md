# Current reference

`CurrentReference` (`current_reference.py`) is a leaf multi-output current-reference source holding a 2-D `[mode][tap]` bank of tap rows: it carries the reference's static PPA and hands per-call tap values to consumers through `CurrentReferenceSnap`, read back via `i_ref__uA`, performing no compute and emitting no dynamic energy or latency. Mode selection is quasi-static — a consumer indexes one mode row and holds it, at no per-conversion energy.

## Design decisions

- **PPA is static-only; no dynamic energy, no latency.** Area and the always-on bias power live in the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. The bias power that generates the reference currents is folded into `leakage_per_inst__uW` and is **not** derived from the tap values. There is no `v_supply__V` field.
- **Two-stage non-ideality split mirrors the fabrication lifecycle.** The initial-accuracy spread is a per-die constant sampled once in `_sample_fabricate_mismatch` (stored in the `i_refs__uA` buffer); the per-read noise is resampled every `snapshot`.
- **One relative (multiplicative) sigma per stage.** A single scalar covers taps of differing magnitude; an absolute departure would instead need a per-tap tuple.
- **`snapshot(shape=...)` expands, then draws noise.** The fabricated per-instance taps are expanded to the requested `shape` (intrinsic trailing `(mode_num, tap_num)`) before the per-call noise draw, so each position samples independently; the consumer passes `(*inst_shape, mode_num, tap_num)`. An empty `shape` leaves the taps at the fabricated `(*inst_shape, mode_num, tap_num)`.

## Contracts & invariants

- **Canonical leaf signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)`. `T__K` is carried for interface parity; the current model does not use it.
- **Buffer lifecycle.** `nominal_i_refs__uA` (shape `(mode_num, tap_num)`) → `i_refs__uA` (shape `(*inst_shape, mode_num, tap_num)`, post-fabricate). Before any `fabricate()` the actual buffer is the broadcast nominal; `fabricate()` resamples the tolerance (or restores the broadcast nominal when `tolerance` is off). Both buffers are non-persistent.
- **`mode_num` / `tap_num` are properties** on config and module (`len(config.i_refs__uA)` / `len(config.i_refs__uA[0])`). They are the source-side interface a consumer queries for the bank geometry; the consumer holds no tap count of its own.
- **Read path is `i_ref__uA(snap)`.** The encapsulated accessor returns the full `(*inst_shape, mode_num, tap_num)` tap tensor from the snap (never the buffer), so the per-call `noise` is always included; a consumer then indexes its mode row (`[..., mode, :]`) and passes the full per-instance tap ladder `(*inst_shape, tap_num)` straight to its ADC, whose last axis is the reference ladder (`n_ref = tap_num`) and whose leading `(*inst_shape,)` broadcasts right-aligned against the input — no per-tap selection or 1-D collapse. The owner — not the consumer — holds the module, calls `snapshot(shape=(*inst_shape, mode_num, tap_num))` and `i_ref__uA(snap)`, and passes the resulting tensor in.
- **Rows are strictly increasing, equal-length, and non-negative**, validated at config time (`>= 1` mode row, equal row lengths, per-row strictly increasing, each tap `>= 0`); a `0` uA first tap is permitted. The field is always an explicit 2-D bank, including for a single mode; a flat TOML array is rejected by deserialization.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference returns the exact nominal taps even with the `tolerance` policy on; call `fabricate()` to draw the per-die spread.
- **`noise` off still clones**, so a consumer never aliases the internal buffer.

---

- **Reference**: [current_reference](../../../reference/primitive/analog/current_reference.md)
- **Implementation**: `neurox/primitive/analog/current_reference.py`
- **Tests**: `tests/primitive/analog/test_current_reference.py`
