# Prober

## Summary

`Prober` (`common/prober.py`) is the context manager that collects the probe side channel: full per-call tensors that hosts emit on named channels, for calibration fits and diagnostics. It is the collector half of the mechanism — the per-module emitter is the probe mixin's `_probe_record` hook. `AdcProber` is the ADC-scoped subclass carrying the calibration channel names (`adc.convert`, `adc.ideal_vmm`) and the paired calibration view a rescale fit consumes.

## Design decisions

- **Tensors arrive on a side channel, not in return values.** Same rationale as the profiler: every numerical return stays the analog / digital result, and a leaf deep in a composite hands its intermediates out without widening any call signature. An emit is a pure side channel — a no-op without an active prober, never altering host numerics.
- **Probers stack; the profiler slot does not.** The profiler holds one active context per thread because events are scalar reductions any single collector aggregates. A probe session is a windowed capture: an outer session-scoped prober keeps recording while an inner one captures a narrow window, so activation is a class-level LIFO stack and an emission reaches every stacked prober.
- **Channel gating is the prober's, not the emitter's.** The emitter's hook is presence-gated only; each prober applies its own `channels` allowlist at `submit`. One emission site can thus feed differently-scoped probers simultaneously, and adding a channel never touches emitter code.
- **Records are stored full, detached, on the recording device.** Unlike the profiler's 0-D reductions, a probe's value is the tensor itself (a fit consumes every element), so `submit` detaches without reducing, syncing, or copying to host. The memory cost is deliberate: a probing run holds every submitted tensor alive until the prober is dropped.
- **A record carries its emitting module, not a name.** A module cannot name itself; `resolve_names(root)` builds a read-only `id(module) -> qualified name` map from a root's traversal, mirroring the profiler's report-side naming.
- **Pairing is positional.** `paired(ch_a, ch_b)` zips two channels record-by-record and rejects a count mismatch. The intended use runs the same stimulus sequence through a physical and an ideal model so record `i` on each channel describes the same call.

## Contracts & invariants

- **Emit at most once per logical operation per channel.** The pairing contract depends on record order matching the call sequence.
- **LIFO discipline.** `__exit__` pops its own frame and raises if the stack top is not `self`; `with`-block usage guarantees this. The stack is class-level and process-wide, not thread-scoped.
- **The allowlist is init-fixed.** `channels=None` accepts every channel; a `frozenset` admits exactly its members. `AdcProber()` defaults to the two ADC channels.

### Public API

- `submit(channel, module, tensors)` — allowlist gate, then store one detached `(module, dict)` record in submission order (called by `ProbeMixin._probe_record`).
- `records(channel)` — submission-ordered record list; empty list for an unseen channel.
- `stacked(channel, key)` — one named tensor stacked across records with a new leading record axis; raises on an empty channel.
- `paired(ch_a, ch_b)` — order-aligned record pairs; raises on a count mismatch.
- `resolve_names(root)` — read-only `id -> qualified name` map from `root.named_modules()`.
- `AdcProber` — `ADC_CONVERT` / `ADC_IDEAL_VMM` constants, ADC-channel default allowlist, and the views `convert_records()` / `ideal_vmm_records()` / `paired_conversions()`.

## Performance & resources

With no active prober `_probe_record` returns after one empty-list check — no tensor is touched — and `@torch.compiler.disable` keeps the hook out of any caller's compiled graph. Per emission with $k$ active probers: $k$ allowlist checks plus, per admitting prober, one detached-view dict and a Python append; no device sync ever happens on the emit path. Memory grows linearly with admitted records.

## Gotchas

- **Do not probe an unbounded run.** Records are full tensors; a long capture accumulates them all. Scope the `with` block to the stimulus batch being fitted.
- **`stacked` requires a common shape.** Records on one channel must carry the key at one shape; mixed-geometry captures must be read via `records` instead.

## Known limitations

- **No persistence.** Records live in process memory only; a calibration tool serializes what it needs.

---

- **Reference**: N/A — the prober has no physics spec; the calibration procedures consuming it live with their tools
- **Implementation**: `neurox/common/prober.py`, `neurox/common/mixin/probe.py`
- **Tests**: `tests/common/test_prober.py`
