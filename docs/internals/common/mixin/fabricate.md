# Fabricate mixin

## Summary

`FabricateMixin` grants a host module an automatic, pre-order `fabricate()` cascade for static manufacturing-variation sampling. A host inherits it alongside `nn.Module` and sets `self._inst_shape` at construction; the inherited `fabricate()` resamples the host's own static state then recurses into every `FabricateMixin` descendant, so a new layered module participates in fabrication for free. Subclasses override only the per-layer sampling step. The cross-cutting four-phase lifecycle in which this cascade sits is described in [physical_state](../../physical_state.md).

## Design decisions

- **One inherited auto-cascade; subclasses override only the sampling step.** The walk is structural recursion over `self.children()` — self first, then each `FabricateMixin` child. The rejected alternative was each parent manually driving its children, which makes every tree refactor edit every parent. Structural recursion means container nodes that own no static state inherit a no-op body and ship no sampling code.
- **The container expansion is built into the cascade, not the subclass.** `nn.ModuleList` and `nn.ModuleDict` children are expanded transparently so their `FabricateMixin` members are reached without the host enumerating them. This keeps hosts that hold lists/dicts of tiles free of cascade boilerplate.
- **Non-`FabricateMixin` children are skipped silently, by design.** Auxiliary `nn.Module` attributes (helpers, `nn.Parameter` holders) coexist in the same tree without participating in or being broken by fabrication, and without any opt-out flag.
- **`_inst_shape` is a host responsibility, declared as a class annotation.** The mixin reads but never assigns it; committing the per-instance multiplicity belongs to the host `__init__`, keeping the mixin shape-agnostic.

## Contracts & invariants

- **Host requirements.** A host must also inherit `nn.Module` (the cascade walks `self.children()`) and must assign `self._inst_shape: tuple[int, ...]` in its `__init__`, encoding the per-instance multiplicity at this layer.
- **Override surface.** Subclasses override `_sample_fabricate_mismatch(self) -> None` to resample their own static state. The default body is a no-op, which is the correct body for cascade-only container nodes that own no static state.
- **`fabricate(self) -> None` semantics.** Pre-order: it samples the host's own mismatch, then recurses into each fabricable child. A child reached through an `nn.ModuleList` / `nn.ModuleDict` is visited as if it were a direct child. Children that are not `FabricateMixin` are not visited.
- **Re-callable without accumulation.** `fabricate()` may be called any number of times. Each call resamples from the host's unchanged nominal template, so no state accumulates across calls; the result depends only on the host's nominal buffers and the RNG draw, not on prior fabricate calls.
- **No weight is touched.** The cascade samples static mismatch only. Writing the programmed weight is a separate, manually dispatched concern (see [physical_state](../../physical_state.md)); the two write orthogonal state and their call order is free.

## Performance & resources

- A `fabricate()` call is a single pre-order traversal of the module subtree — cost linear in node count. Per-node resampling reads the (small, usually 0-d) nominal buffers, so allocation scales with each host's `inst_shape`, not with any per-call batch.

## Gotchas

- **Sampling uses nondeterministic RNG.** The bodies invoked through `_sample_fabricate_mismatch` draw random mismatch; `torch.use_deterministic_algorithms(True)` conflicts with this, and under DDP each rank draws its own realisation (the cascade does not synchronise RNG across ranks).
- **Skipping `fabricate()` is silent.** A host whose `fabricate()` was never called keeps its nominal (mismatch-free) state rather than raising; a surprisingly clean result can mean the cascade was never driven.
- **The cascade follows the module tree, not arbitrary attributes.** A `FabricateMixin` held outside `self.children()` (e.g. stored only in a plain Python list, not an `nn.ModuleList`) is not reached. Register fabricable submodules as modules or inside `nn.ModuleList` / `nn.ModuleDict`.

## Known limitations

- N/A — the mixin is a complete, self-contained cascade mechanism; whole-lifecycle limitations live in [physical_state](../../physical_state.md).

---

- **Reference**: N/A — software mechanism, no physics reference twin.
- **Implementation**: `neurox/common/mixin/fabricate.py`
- **Tests**: `tests/test_signal_chain.py`, `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this mixin; lifecycle-level decisions are in [physical_state](../../physical_state.md).
