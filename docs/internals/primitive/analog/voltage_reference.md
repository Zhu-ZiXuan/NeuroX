# Voltage reference

## Design decisions

- **PPA is static-only; nothing to log per call.** The reference emits no dynamic energy and no latency, so its area and always-on bias power ride the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. There is no `v_supply__V` field and no bias-current field.
- **Two-stage sampling split.** The initial-accuracy spread is sampled once in `_sample_fabricate_mismatch` and held as ordinary `v_refs__V` fabricated state; the per-read noise is resampled every `snapshot`. This fabricate-then-runtime split holds a die's tolerance fixed across reads while the per-read noise varies per call.
- **Both departures are relative (multiplicative).** `nominal * (1 + randn * sigma)` for each stage, so one sigma applies uniformly to taps of differing magnitude; an absolute per-tap sigma would need a tuple.
- **`snapshot()` takes no external `shape`.** A reference has no per-element solve grid to broadcast onto; its output is intrinsically `(*inst_shape, num_refs)`.

## Contracts & invariants

- **State lifecycle.** `nominal_v_refs__V` is a non-persistent source buffer with shape `(num_refs,)`. `fabricate()` expands it to `(*inst_shape, num_refs)` and assigns `v_refs__V` as ordinary fabricated state, with tolerance applied when enabled. Snapshotting requires fabrication first.
- **`num_refs` is a property** = `len(config.v_refs__V)` — an init-determined constant. It is the source-side interface for how many taps exist.
- **Read path is `v_ref__V(snap)`.** The accessor returns the full `(*inst_shape, num_refs)` tap tensor from the snap rather than the buffer, so the per-call noise is always included.
- **Config validation.** `validate_taps` requires `>= 1` tap, each `>= 0` (ordering is not enforced); `validate_noise` requires both sigmas `>= 0`.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference has no `v_refs__V` state; call `fabricate()` before snapshotting.
- **`noise` off still clones.** `snapshot` returns a fresh tensor either way.

---

- **Reference**: [voltage_reference](../../../reference/primitive/analog/voltage_reference.md)
- **Implementation**: `neurox/primitive/analog/voltage_reference.py`
- **Tests**: `tests/primitive/analog/test_voltage_reference.py`
