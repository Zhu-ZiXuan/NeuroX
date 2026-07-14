# Profile mixin

## Summary

`ProfileMixin` is the per-module PPA surface: it gives a host module an emitter identity — a hierarchical dotted instance name plus a class-derived type tag — two emit hooks, `_log_dynamic_energy` and `_log_latency`, through which a leaf attributes its own runtime energy and latency to the active profiler, and the static-PPA property contract (`area__um2` / `leakage__uW`) the profiler's static walk reads. The mixin is physics-agnostic — it carries no PPA values, no thresholds, and no awareness of how any quantity is computed; the two static properties are concrete aggregators that scale the per-instance data a sized leaf sets by `inst_count`, and the emit hooks only route a caller-built tensor to the collector. The collector half — event capture, batched sync, aggregation, and the static walk — lives in the profiler.

## Design decisions

- **Name is owner-constructed, never self-derived.** A host receives its `name` from the parent that builds it; the parent composes a dotted hierarchical path (a child built under a prefix appears as `parent.child`), so the same leaf type appears under distinct qualified names per placement. The mixin stores that string and exposes it read-only as `qualified_name`; it never inspects the module tree or guesses a name from its class. Only the owner knows the instance's role in the hierarchy, and a self-derived name could not disambiguate two siblings of the same type; the class-name `module_type` tag emitted alongside is self-derived precisely because class identity — unlike placement — is unambiguous.
- **Two independent emit hooks, never one fused PPA call.** Energy and latency are separate runtime quantities; a leaf may own a physical model for one and not the other. Keeping `_log_dynamic_energy` and `_log_latency` separate lets each call site gate each quantity on its own config, instead of forcing a caller to fabricate a zero for the quantity it does not model. A fused `log(energy, latency)` entry was rejected for that reason.
- **No value-based gating inside the hooks.** The hooks route a tensor and nothing else; the "should I emit?" decision belongs to the leaf, made at the call site from its own config. This keeps the mixin free of PPA fields and thresholds and is what lets a latency-only or energy-only leaf express itself by simply not calling the other hook. Pushing a zero tensor through unconditionally would record a spurious event, so a leaf that wants conditional emission guards the call itself.
- **Both quantities are passed as tensors, not Python floats.** A leaf hands over a per-element energy map or a 0-D latency scalar at the hook; the batched device→host sync that this tensor form enables is the collector's.
- **The mixin owns the static-PPA aggregation.** `area__um2` and `leakage__uW` are concrete `@property` on the mixin, each returning `self._<field>_per_inst__* × inst_count`; a sized leaf sets the bare per-instance data (`_area_per_inst__um2` / `_leakage_per_inst__uW`) in its `__init__`, usually from its config's `*_per_inst__*` field, though a leaf may set a fixed value instead. The profiler's static walk keys on `isinstance(m, ProfileMixin)` and reads the two properties, so composing `ProfileMixin` is the single act that both enables emission and enrols a module in the static walk — and composing it without setting the per-instance data raises `AttributeError` on the first static-PPA read, a loud failure rather than a silent zero. A module that owns no reportable silicon (a device, the current mirror / mux, an xbar cell) composes no `ProfileMixin` and stays outside the walk entirely; a composite reports only its own overhead, its children self-report.
- **Each emit is independent; the mixin accumulates no per-call state.** Cross-call aggregation is the collector's. A leaf calls each hook at most once per logical operation, and a composite emits only its own per-op overhead — never a sum of its children's, since per-event aggregation happens collector-side.
- **The emit hooks are `@torch.compiler.disable`.** They touch host state the compiler cannot trace, so they sit outside any compiled region. The break is local, taken after the kernel math, so in-kernel fusion is unaffected — see [compile/contracts](../../compile/contracts.md).

## Composition

The mixin composes alongside `nn.Module` and carries no auto-fired hook, so its MRO position relative to the other circuit mixins is not order-sensitive. Its trigger point is the end of a leaf's primary method: a leaf calls `_log_dynamic_energy` / `_log_latency` after all kernel math, once the output tensor exists.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/profile.py`
- **Tests**: `tests/test_current_readout_energy.py`, `tests/test_reference_sources.py`
