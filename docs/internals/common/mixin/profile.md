# Profile mixin

## Summary

`ProfileMixin` is the per-module emitter of the PPA side channel: it gives a host module a hierarchical dotted instance name plus two emit hooks, `_log_dynamic_energy` and `_log_latency`, through which a leaf attributes its own runtime energy and latency to the active profiler. The mixin is physics-agnostic — it carries no PPA fields, no thresholds, and no awareness of how either quantity is computed; it only routes a caller-built tensor to the collector. The collector half — event capture, batched sync, aggregation, and the static walk — lives in [profiler](../profiler.md).

## Design decisions

- **Name is owner-constructed, never self-derived.** A host receives its `name` from the parent that builds it; the parent composes a dotted hierarchical path (a child built under a prefix appears as `parent.child`), so the same leaf type appears under distinct qualified names per placement. The mixin stores that string and exposes it read-only as `qualified_name`; it never inspects the module tree or guesses a name from its class. Only the owner knows the instance's role in the hierarchy, and a self-derived name could not disambiguate two siblings of the same type.
- **Two independent emit hooks, never one fused PPA call.** Energy and latency are separate runtime quantities; a leaf may own a physical model for one and not the other. Keeping `_log_dynamic_energy` and `_log_latency` separate lets each call site gate each quantity on its own config, instead of forcing a caller to fabricate a zero for the quantity it does not model. A fused `log(energy, latency)` entry was rejected for that reason.
- **No value-based gating inside the hooks.** The hooks route a tensor and nothing else; the "should I emit?" decision belongs to the leaf, made at the call site from its own config. This keeps the mixin free of PPA fields and thresholds and is what lets a latency-only or energy-only leaf express itself by simply not calling the other hook. Pushing a zero tensor through unconditionally would record a spurious event, so a leaf that wants conditional emission guards the call itself.
- **Both quantities are passed as tensors, not Python floats.** A leaf hands over a per-element energy map or a 0-D latency scalar at the hook; the batched device→host sync that this tensor form enables is the collector's, in [profiler](../profiler.md).
- **The mixin owns no static PPA surface.** Area and leakage live on the circuit layer, not on the mixin — an orchestration macro emits dynamic events and has a name yet owns no silicon, so it contributes nothing to the static walk. That static walk keys on the circuit layer and is the collector's; see [profiler](../profiler.md) and [`ADR-0002`](../../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md).
- **Each emit is independent; the mixin accumulates no per-call state.** Cross-call aggregation is the collector's. A leaf calls each hook at most once per logical operation, and a composite emits only its own per-op overhead — never a sum of its children's, since per-event aggregation happens collector-side.
- **The emit hooks are `@torch.compiler.disable`.** They touch host state the compiler cannot trace, so they sit outside any compiled region. The break is local, taken after the kernel math, so in-kernel fusion is unaffected and library code runs with `fullgraph=False` — see [compile/contracts](../../compile/contracts.md). A tensor-return channel that would remove this single graph break is an open option tracked in [compile](../../compile/README.md); the side-channel break is accepted for now.

## Composition

The mixin composes alongside `nn.Module` and carries no auto-fired hook, so its MRO position relative to the other circuit mixins is not order-sensitive. Its trigger point is the end of a leaf's primary method: a leaf calls `_log_dynamic_energy` / `_log_latency` after all kernel math, once the output tensor exists.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/profile.py`
- **Tests**: `tests/test_current_readout_energy.py`, `tests/test_reference_sources.py`
- **Decisions**: N/A — emitter mechanism; the device-vs-circuit split it relies on is in [`ADR-0002`](../../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
