# ProfileMixin

## Summary

`ProfileMixin` is the per-module emitter of the PPA side channel: it gives any module a hierarchical dotted instance name plus two emit hooks, `_log_dynamic_energy` and `_log_latency`, through which a leaf attributes its own runtime energy and latency to the active profiler. The mixin is physics-agnostic — it carries no PPA fields, no thresholds, and no awareness of how either quantity is computed; it only routes a caller-built tensor to the collector. The collector half — event capture, batched sync, aggregation, static walk — lives in [profiler](../profiler.md).

## Design decisions

- **Name is owner-constructed, never self-derived.** A module receives its `name` from the parent that builds it; the parent composes a dotted hierarchical path (a child built under a prefix appears as `parent.child`), so the same leaf type appears under distinct qualified names per placement. The mixin stores that string and exposes it read-only as `qualified_name`; it never inspects the module tree or guesses a name from its class. Rationale: only the owner knows the instance's role in the hierarchy, and a self-derived name could not disambiguate two siblings of the same type.
- **Two independent emit hooks, never one fused PPA call.** Energy and latency are separate runtime quantities; a leaf may own a physical model for one and not the other. Keeping `_log_dynamic_energy` and `_log_latency` separate lets each call site gate each quantity on its own config, instead of forcing a caller to fabricate a zero for the quantity it does not model. A fused `log(energy, latency)` entry was rejected for that reason.
- **No value-based gating inside the hooks.** The hooks route a tensor and nothing else; the "should I emit?" decision belongs to the leaf, made at the call site from its own config. This keeps the mixin free of PPA fields and thresholds and is what lets a latency-only or energy-only leaf express itself by simply not calling the other hook.
- **Both quantities are passed as tensors, not Python floats.** A leaf hands over a per-element energy map or a 0-D latency scalar at the hook; the batched device→host sync that this tensor form enables is the collector's, in [profiler](../profiler.md).
- **The mixin owns no static PPA surface.** Area and leakage live on the circuit layer, not on the mixin — an orchestration macro (a `ProfileMixin` but not a `CircuitBase`) has a name and may emit dynamic events but owns no silicon. The static walk that keys on the circuit layer is the collector's; see [profiler](../profiler.md) and [`ADR-0002`](../../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md).

## Contracts & invariants

### Public API

- `__init__(self, name: str) -> None` — stores the owner-supplied instance name. The mixin does not validate or transform the string; the dotted-path convention is the parent's responsibility.
- `qualified_name -> str` (property) — the hierarchical dotted instance name. Property because it is fixed at construction. Carried as the emitter identity in every event.
- `module_type -> str` (property) — the short class-name tag of the concrete subclass, emitted alongside the name so the collector can group by type as well as by name.
- `_log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None` — record one dynamic-energy event to the active profiler; no-op when none is active. The tensor is the leaf's per-op switching energy, in fJ, built caller-side from the leaf's physical model.
- `_log_latency(self, latency__ns: Tensor) -> None` — record one latency event to the active profiler; no-op when none is active. The tensor is the leaf's per-op latency contribution, in ns, built caller-side as per-op latency times a serial-op count.

### Invariants

- **`@torch.compiler.disable` on both emit hooks.** The hooks touch host state the compiler cannot trace, so they stay outside any compiled region; the break is local, taken at the end of a leaf's primary method after all kernel math, so in-kernel fusion is unaffected. Library code therefore runs with `fullgraph=False` — see [compile/contracts](../../compile/contracts.md).
- **No-op outside an active profiler.** Each hook queries the thread-scoped active profiler; if none is active on the calling thread it returns without effect. Emission and the enclosing collection context must run on the same thread.
- **Emit exactly once per logical operation, at the call site.** A leaf calls each hook at most once per forward; a composite emits only its own per-op overhead and does not sum its children's emissions — each child with a dynamic model emits directly, and per-event aggregation happens collector-side. Summing at the composite would double-count.
- **Tensor in, nothing back.** The hooks return `None`; the numerical result of a forward travels the return value, never these hooks. PPA quantities ride a side channel so the hot-path signature stays narrow.
- **Caller owns tensor construction.** The leaf builds `dynamic_energy__fJ` / `latency__ns` itself (e.g. `full_like` the output for per-element energy, a 0-D tensor for latency) on the recording device and dtype; the mixin does not allocate or reshape.

## Performance & resources

Per emit call: one thread-local lookup and a single delegation; the mixin allocates nothing of its own, and `qualified_name` / `module_type` are pure field / `type(self).__name__` reads. The deferred, batched device→host sync that keeps total host-sync cost independent of emit count is the collector's, in [profiler](../profiler.md).

## Gotchas

- **Do not gate emission inside the hook.** The hook performs no value check; a leaf that wants conditional emission must guard the call itself (e.g. only call `_log_latency` when its config declares a non-zero per-op latency). Pushing a zero tensor through unconditionally records a spurious event.
- **The name is not self-validating.** Because the mixin stores the owner-supplied string verbatim, a parent that omits or duplicates a prefix produces colliding qualified names, silently merging two emitters in by-name aggregation.
- **`module_type` is the concrete class, not a configured label.** It is `type(self).__name__`, so renaming a class changes the grouping key; it is not a stable identifier a report consumer can pin.

## Known limitations

- **No tensor-return profiling path.** A return-value channel that removes the single graph break is an open option (tracked in [compile](../../compile/README.md)); the side-channel break is accepted for now.
- **Serial-op multiplicity is the leaf's, not the mixin's.** The mixin only requires a tensor; how a leaf derives the serial-op count that scales its latency tensor is part of each leaf's cost model and is specified in that leaf's reference page.

---

- **Reference**: N/A — the emitter is a software mechanism; the per-quantity PPA cost models live in `reference/<subsystem>/`.
- **Implementation**: `neurox/common/mixin/profile.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_readout_log_gating.py`
- **Decisions**: N/A — emitter mechanism; the device-vs-circuit split it relies on is in [`ADR-0002`](../../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
