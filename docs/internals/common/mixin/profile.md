# ProfileMixin — Implementation

## Summary

`ProfileMixin` is the per-module half of the PPA collection mechanism: it gives any module a hierarchical dotted instance name plus two side-channel emit hooks, `_log_dynamic_energy` and `_log_latency`, through which a leaf attributes its own runtime energy and latency to the active profiler. The mixin is physics-agnostic — it carries no PPA fields, no thresholds, and no awareness of how either quantity is computed; it only routes a caller-built tensor to the collector. The collector side (event capture, batched sync, aggregation, static walk) lives in [profiler](../profiler.md); this document covers the emitter contract only.

## Design decisions

- **Name is owner-constructed, never self-derived.** A module receives its `name` from the parent that builds it; the parent composes a dotted hierarchical path (a child built under a prefix appears as `parent.child`), so the same leaf type appears under distinct qualified names per placement. The mixin stores that string and exposes it read-only as `qualified_name`; it never inspects the module tree or guesses a name from its class. Rationale: only the owner knows the instance's role in the hierarchy, and a self-derived name could not disambiguate two siblings of the same type.
- **Two independent emit hooks, never one fused PPA call.** Energy and latency are separate runtime quantities; a leaf may own a physical model for one and not the other. Keeping `_log_dynamic_energy` and `_log_latency` separate lets each call site gate each quantity on its own config, instead of forcing a caller to fabricate a zero for the quantity it does not model. A fused `log(energy, latency)` entry was rejected for that reason.
- **No value-based gating inside the hooks.** The hooks route a tensor and nothing else; the "should I emit?" decision belongs to the leaf, made at the call site from its own config. This keeps the mixin free of PPA fields and thresholds and is what lets a latency-only or energy-only leaf express itself by simply not calling the other hook.
- **Both quantities are passed as tensors, not Python floats.** Energy is genuinely per-element (per-instance switching energy that the collector reduces); latency is a per-op constant times a runtime serial-op count, packaged as a 0-D tensor so it shares the energy path's deferred GPU→CPU sync rather than forcing a per-event host sync on one float.
- **The mixin owns no PPA surface.** Static area / leakage live on the circuit layer, not here — an orchestration macro (a `ProfileMixin` but not a `CircuitBase`) has a name and may emit dynamic events but owns no silicon. This split is why the collector's static walk keys on the circuit layer, not on `ProfileMixin`.

## Contracts & invariants

### Public API

- `__init__(self, name: str) -> None` — stores the owner-supplied instance name. The mixin does not validate or transform the string; the dotted-path convention is the parent's responsibility.
- `qualified_name -> str` (property) — the hierarchical dotted instance name. Property because it is fixed at construction. Included as the emitter identity in every event.
- `module_type -> str` (property) — the short class-name tag of the concrete subclass, emitted alongside the name so the collector can group by type as well as by name.
- `_log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None` — record one dynamic-energy event to the active profiler; no-op when none is active. The tensor is the leaf's per-op switching energy, in fJ, built caller-side from the leaf's physical model.
- `_log_latency(self, latency__ns: Tensor) -> None` — record one latency event to the active profiler; no-op when none is active. The tensor is the leaf's per-op latency contribution, in ns, built caller-side as per-op latency times a serial-op count.

### Invariants

- **`@torch.compiler.disable` on both emit hooks.** The collector reads a thread-local slot and mutates a Python list — both untraceable by dynamo — so the hooks must stay outside the compiled graph. The break is local: each hook is called at the end of a leaf's primary method, after all kernel math, so in-kernel fusion is unaffected. Library code therefore cannot assume `fullgraph=True`. See [compile/contracts](../../compile/contracts.md).
- **No-op outside an active profiler.** Each hook queries the current profiler; if none is active on the calling thread it returns without effect. The active profiler is thread-scoped, so emission and the enclosing collection context must run on the same thread.
- **Emit exactly once per logical operation, at the call site.** A leaf calls each hook at most once per forward; a composite emits only its own per-op overhead and does not sum its children's emissions — each child with a dynamic model emits directly, and the collector's event aggregation is the single source of truth. Summing at the composite would double-count.
- **Tensor in, nothing back.** The hooks return `None`; the numerical result of a forward travels the return value, never these hooks. PPA quantities ride a side channel so the hot-path signature stays narrow.
- **Caller owns tensor construction.** The leaf builds `dynamic_energy__fJ` / `latency__ns` itself (e.g. `full_like` the output for per-element energy, a 0-D tensor for latency) on the recording device and dtype; the mixin does not allocate or reshape.

## Performance & resources

Per emit call: one thread-local lookup, then a single delegation into the collector (which stashes a reduced 0-D tensor and appends one Python entry). No host sync occurs at the hook — the GPU→CPU drain is deferred and batched by the collector at context exit, so total host-sync cost is independent of emit count. The mixin allocates nothing of its own; `qualified_name` and `module_type` are pure field / `type(self).__name__` reads.

## Gotchas

- **A graph break is expected at every emit site.** The `@torch.compiler.disable` boundary means a leaf's primary method cannot compile as one `fullgraph` region. Treat the break as designed; reserve `fullgraph=True` for tests that want to catch an accidental Python sync elsewhere.
- **Do not gate emission inside the hook.** The hook performs no value check; a leaf that wants conditional emission must guard the call itself (e.g. only call `_log_latency` when its config declares a non-zero per-op latency). Pushing a zero tensor through unconditionally records a spurious event.
- **The name is not self-validating.** Because the mixin stores the owner-supplied string verbatim, a parent that omits or duplicates a prefix produces colliding qualified names; the collector groups by that string, so a collision silently merges two emitters in by-name aggregation.
- **`module_type` is the concrete class, not a configured label.** It is `type(self).__name__`, so renaming a class changes the grouping key; it is not a stable identifier you can pin in a report consumer.

## Known limitations

- **No tensor-return profiling path.** A return-value channel that removes the single graph break is an open option (tracked in [compile](../../compile/README.md)); the side-channel break is accepted for now.
- **Serial-op multiplicity is out of scope here.** The mixin only requires a tensor; how a leaf derives the serial-op count that scales its latency tensor is part of each leaf's cost model and is specified in that leaf's reference page, not here.

---

- **Reference**: N/A — the emitter is a software mechanism; the per-quantity PPA cost models live in `reference/<subsystem>/`.
- **Implementation**: `neurox/common/mixin/profile.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_readout_log_gating.py`
- **Decisions**: N/A — emitter mechanism; the device-vs-circuit split it relies on is in [`ADR-0002`](../../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
