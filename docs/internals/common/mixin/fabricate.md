# Fabricate mixin

## Summary

`FabricateMixin` grants a host module an automatic, pre-order `fabricate()` cascade for static manufacturing-variation sampling. A host inherits it alongside `nn.Module` and sets `self._inst_shape` at construction; the inherited `fabricate()` resamples the host's own static state then recurses into every `FabricateMixin` descendant, so a new layered module participates in fabrication for free. Subclasses override only the per-layer sampling step. The cross-cutting four-phase lifecycle in which this cascade sits is described in [physical_state](../../physical_state.md).

## Design decisions

- **One inherited auto-cascade; each subclass supplies only its own sampling step.** The walk is structural recursion over `self.children()` — self first, then each `FabricateMixin` child. The rejected alternative was each parent manually driving its children, which makes every tree refactor edit every parent.
- **No root default.** The mixin ships no default no-op, so a node that never declares its own fabrication behavior fails loudly instead of silently sampling nothing.
- **The container expansion is built into the cascade, not the subclass.** `nn.ModuleList` and `nn.ModuleDict` children are expanded transparently so their `FabricateMixin` members are reached without the host enumerating them. This keeps hosts that hold lists/dicts of tiles free of cascade boilerplate.
- **Non-`FabricateMixin` children are skipped silently, by design.** Auxiliary `nn.Module` attributes (helpers, `nn.Parameter` holders) coexist in the same tree without participating in or being broken by fabrication, and without any opt-out flag.
- **`_inst_shape` is a host responsibility, declared as a class annotation.** The mixin reads but never assigns it; committing the per-instance multiplicity belongs to the host `__init__`, keeping the mixin shape-agnostic.
- **Repeated `fabricate()` calls do not accumulate.** Each call resamples from the host's unchanged nominal template, so the result depends only on the nominal buffers and the RNG draw, never on prior calls.

## Composition

The mixin declares no `__init__` and overrides no `nn.Module` method, so it imposes no constructor-chaining order and composes with `nn.Module` and any sibling mixin freely. Within the physical-state lifecycle it runs after `__init__`, before inference, and in either order with `program(...)` (see [physical_state](../../physical_state.md)).

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/fabricate.py`
- **Tests**: `tests/test_signal_chain.py`, `tests/test_reference_sources.py`
