# Voltage reference

## Summary

`VoltageReference` (`voltage_reference.py`) is a leaf multi-output voltage reference source: it holds a tuple of nominal voltage taps, perturbs them with a fabricate-time tolerance and a per-read noise, and hands the result to consumers through `VoltageReferenceSnap`, read back via the `v_ref__V` accessor. It computes nothing and emits no dynamic energy or latency — its hardware cost is entirely static.

## Design decisions

- **PPA is static-only; no dynamic energy, no latency.** The taps are ideal high-impedance nodes that draw no signal current, so there is nothing to log per call. Area and the always-on bias power live in the inherited `area_per_inst__um2` / `leakage_per_inst__uW`, collected by the profiler's static walk over `CircuitBase` — the same stance as voltage_driver. There is no `v_supply__V` field and no bias-current field.
- **Two-stage non-ideality split mirrors the fabrication lifecycle.** The initial-accuracy spread is a per-die constant, so it is sampled once in `_sample_fabricate_mismatch` and stored in the `v_refs__V` buffer; the per-read noise is resampled every `snapshot`. This is the switch_cap-style fabricate-then-runtime split, not the per-call-only resample some other leaves use — a reference's tolerance must not change between two reads of the same die.
- **Both departures are relative (multiplicative).** `nominal * (1 + randn * sigma)` for each stage, so one sigma applies uniformly to taps of differing magnitude; an absolute per-tap sigma would need a tuple. This matches `CurrentMirror`'s relative ratio mismatch.
- **`snapshot()` takes no external `shape`.** Unlike the clamp drivers, a reference has no per-element solve grid to broadcast onto; its output is intrinsically `(*inst_shape, num_refs)`. A consumer selects a tap and broadcasts it onto its own grid.

## Contracts & invariants

- **Canonical leaf signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)`. `T__K` is carried for interface parity; the current model does not use it.
- **Buffer lifecycle.** `nominal_v_refs__V` (shape `(num_refs,)`, design intent) → `v_refs__V` (shape `(*inst_shape, num_refs)`, post-fabricate actual). Before any `fabricate()` the actual buffer is the broadcast nominal; `fabricate()` resamples the tolerance (or restores the broadcast nominal when `tolerance` is off). Both buffers are non-persistent.
- **`num_refs` is a property** = `len(config.v_refs__V)` — an init-determined constant. It is the source-side interface a consumer queries for how many taps exist; the consumer holds no tap count of its own.
- **Read path is `v_ref__V(snap)`.** The encapsulated accessor returns the full `(*inst_shape, num_refs)` tap tensor from the snap (never the buffer), so the per-call `noise` is always included; a consumer then selects a tap by index and broadcasts it. The owner — not the consumer — holds the module, calls `snapshot()` once and `v_ref__V(snap)`, and passes the resulting tensor in.
- **Taps are unordered and non-negative**, validated at config time (`>= 1` tap, each `>= 0`); a `0` V tap is valid and denotes a ground/rail reference (relative noise `* 0 == 0`, so it stays stable and exact). There is no ordering requirement on the taps.

## Performance & resources

N/A — construction-time buffer fills plus per-call elementwise samples, off any memory- or compile-critical path.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference returns the exact nominal taps even with the `tolerance` policy on; call `fabricate()` to draw the per-die spread.
- **`noise` off still clones.** `snapshot` returns a fresh tensor either way, so a consumer never aliases the internal buffer.

## Known limitations

- No load-dependent droop (ideal high-impedance taps) and no correlated temperature drift — the per-read term is i.i.d. noise, not slow drift.

---

- **Reference**: [voltage_reference](../../reference/analog/voltage_reference.md)
- **Implementation**: `neurox/analog/voltage_reference.py`
- **Tests**: `tests/test_reference_sources.py`
