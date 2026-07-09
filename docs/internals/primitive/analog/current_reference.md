# Current reference

`CurrentReference` (`current_reference.py`) is a leaf multi-output current-reference source: it carries the reference's static PPA and hands per-call tap values to consumers through `CurrentReferenceSnap`, read back via `i_ref__uA`, performing no compute and emitting no dynamic energy or latency.

## Design decisions

- **PPA is static-only; no dynamic energy, no latency.** Area and the always-on bias power live in the inherited `area_per_inst__um2` / `leakage_per_inst__uW`, collected by the profiler's static walk over `CircuitBase`. The bias power that generates the reference currents is folded into `leakage_per_inst__uW` and is **not** derived from the tap values. There is no `v_supply__V` field.
- **Two-stage non-ideality split mirrors the fabrication lifecycle.** The initial-accuracy spread is a per-die constant sampled once in `_sample_fabricate_mismatch` (stored in the `i_refs__uA` buffer); the per-read noise is resampled every `snapshot`.
- **One relative (multiplicative) sigma per stage.** A single scalar covers taps of differing magnitude; an absolute departure would instead need a per-tap tuple.
- **`snapshot()` takes no external `shape`** — the output is intrinsically `(*inst_shape, num_refs)`; a consumer selects a tap and broadcasts it.

## Contracts & invariants

- **Canonical leaf signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)`. `T__K` is carried for interface parity; the current model does not use it.
- **Buffer lifecycle.** `nominal_i_refs__uA` (shape `(num_refs,)`) → `i_refs__uA` (shape `(*inst_shape, num_refs)`, post-fabricate). Before any `fabricate()` the actual buffer is the broadcast nominal; `fabricate()` resamples the tolerance (or restores the broadcast nominal when `tolerance` is off). Both buffers are non-persistent.
- **`num_refs` is a property** = `len(config.i_refs__uA)`. It is the source-side interface a consumer queries for how many taps exist; the consumer holds no tap count of its own.
- **Read path is `i_ref__uA(snap)`.** The encapsulated accessor returns the full `(*inst_shape, num_refs)` tap tensor from the snap (never the buffer), so the per-call `noise` is always included; a consumer then selects a tap by index and broadcasts it. The owner — not the consumer — holds the module, calls `snapshot()` once and `i_ref__uA(snap)`, and passes the resulting tensor in.
- **Taps are unordered and non-negative**, validated at config time (`>= 1` tap, each `>= 0`); a `0` uA tap is permitted.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference returns the exact nominal taps even with the `tolerance` policy on; call `fabricate()` to draw the per-die spread.
- **`noise` off still clones**, so a consumer never aliases the internal buffer.

---

- **Reference**: [current_reference](../../../reference/primitive/analog/current_reference.md)
- **Implementation**: `neurox/analog/current_reference.py`
- **Tests**: `tests/test_reference_sources.py`
