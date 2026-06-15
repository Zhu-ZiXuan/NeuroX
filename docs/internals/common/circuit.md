# CircuitBase — Implementation

## Summary

`CircuitBase` is the fixed base every electrical-circuit module inherits: the single composition `FabricateMixin + ProfileMixin + nn.Module + Generic[ConfigT]`, plus a static-PPA property surface (`area_per_inst__um2`, `leakage_per_inst__uW`, `inst_count`, `inst_area__um2`, `inst_leakage__uW`) read directly from `self.config`. It is cross-cutting software structure with no physics-bearing reference counterpart; the per-instance area / leakage numbers it exposes are specified per subsystem under [reference](../../reference/README.md).

## Design decisions

- **A fixed base class, not a loose mixin stack.** `CircuitBase` pins the canonical composition `FabricateMixin + ProfileMixin + nn.Module` in that order and adds `Generic[ConfigT]` so the typed config narrows in each leaf. Every electrical circuit needs exactly this set, so naming the composition once removes the per-leaf boilerplate of restating four bases and keeps the order — which determines MRO and `__init__` chaining — uniform. Rejected — leaving each leaf to compose the mixins by hand: the order would drift and the typed-config narrowing would have to be re-declared everywhere.
- **Static PPA stays on this base, not in a separate mixin.** The area / leakage surface lives on `CircuitBase` rather than a standalone `StaticPpaMixin`. Splitting it out would create a load-bearing implicit contract — the profile side would read PPA fields it does not declare — and force each leaf to compose yet another mixin. Folding the surface into the base that already owns the typed config (the source of those numbers) avoids both costs.
- **The typed area / leakage surface is itself the profiler's static-collection predicate.** `CircuitBase` is the single layer that carries the config-backed area / leakage properties, so an `isinstance(m, CircuitBase)` test is exactly "owns static silicon". An orchestration macro (a `ProfileMixin`, but also `FabricateMixin` / `RegistryMixin` / `nn.Module` — not a `CircuitBase`) has a name and may emit dynamic events but owns no static surface, and a device is not a `CircuitBase` at all — both are correctly excluded from the static walk. See [`profiler.md`](profiler.md).
- **No PPA cache.** The derived aggregates (`inst_count`, `inst_area__um2`, `inst_leakage__uW`) are recomputed on every access from `inst_shape` and the per-instance config values; nothing is memoized. They are cheap arithmetic, the static walk reads each at most once per report, and a cache would add an invalidation surface for no measurable saving. Rejected — caching on construction: it would have to track config / shape mutation it cannot observe.

## Contracts & invariants

- **Construction signature.** `__init__(self, *, config: ConfigT, name: str, inst_shape: tuple[int, ...])` is keyword-only. It drives `nn.Module` and `ProfileMixin` setup (`name` is the hierarchical profiler name) and binds `config` and the per-instance fabrication shape. A leaf's own `__init__` calls `super().__init__(config=..., name=..., inst_shape=...)` first, then registers its buffers and extra fields. `FabricateMixin` contributes no initializer and passes through the chain.
- **`config` is the typed PPA source.** `config: ConfigT` is bound to a frozen dataclass whose type inherits `CircuitConfig`, which supplies the two base fields `area_per_inst__um2` and `leakage_per_inst__uW`. The generic parameter (`Driver(CircuitBase[DriverConfig])`) narrows `self.config` so subclass fields type-check without a per-leaf forward declaration; an `nn.Module` whose `__getattr__` blurs the narrowed type may still carry a one-line class-level `config: <LeafConfig>` annotation.
- **Property surface — what is init-determined.** `area_per_inst__um2` and `leakage_per_inst__uW` forward the like-named config fields. `inst_shape` returns the construction-time shape. `inst_count` is the product of `inst_shape`; `inst_area__um2` and `inst_leakage__uW` are the per-instance value times `inst_count`. All are properties because each is a function of init-fixed inputs (config + shape); none has a setter.
- **Per-op latency is not a base concern.** `CircuitConfig` carries area and leakage only — there is no base `latency_per_op__ns` and no base dynamic-energy field. A leaf with a dynamic profile model emits its own per-op latency / energy through the `ProfileMixin` side channel at the end of its primary method; fixed-latency leaves declare `latency_per_op__ns` on their own config, parametric leaves derive it from runtime parameters, and dynamics-less leaves emit nothing (their cost folds into the owning circuit's tensors). The emission mechanism is in [`profiler.md`](profiler.md).
- **Membership boundary.** `CircuitBase` is for electrical circuits that own a config-backed per-instance static cost. Devices (physical primitives) are not `CircuitBase` — their cost rolls up into the owning circuit's `CircuitConfig`, and their config does not inherit `CircuitConfig`. Orchestration macros are not `CircuitBase` either — they own no silicon of their own and their PPA surfaces only through their constituent circuits. See the construction model in [`config_and_construction.md`](../config_and_construction.md).

## Performance & resources

The derived properties are constant-time field reads plus one `prod` over `inst_shape`; no tensor allocation, no host sync. Because nothing is cached, repeated access is repeated arithmetic — negligible against the static walk that reads each property at most once per report. The base adds no per-forward overhead: it contributes structure and accessors only, not hot-path code.

## Gotchas

- **`inst_count` counts fabrication multiplicity, not serial operations.** It is the product of `inst_shape` — the parallel instance count behind one circuit. The serial-op multiplicity a leaf uses to scale latency is a separate, forward-tensor-shape quantity derived at the leaf's emit site; do not read `inst_count` as an op count.
- **Reading `inst_area__um2` is not the subtree total.** It is this circuit's own area times its own multiplicity. The area of a composite's subtree is the profiler's static-walk sum over every `CircuitBase` in the tree, not a field on the parent. The same holds for `inst_leakage__uW`.
- **Do not move device or macro PPA onto a `CircuitConfig`.** A device's static cost belongs to the owning circuit's config; a macro owns none. Making either inherit `CircuitConfig` would put it on the `isinstance(m, CircuitBase)` static walk and double-count or mis-attribute silicon.

## Known limitations

- **TODO — no dedicated test module.** `CircuitBase` carries no behavior of its own beyond accessors; the static-collection predicate it anchors is exercised through the profiler tests, and there is no standalone `tests/test_circuit.py`.

---

- **Reference**: N/A — cross-cutting base; per-instance area / leakage numbers are specified per subsystem under [reference](../../reference/README.md)
- **Implementation**: `neurox/common/circuit.py`
- **Tests**: TODO — covered indirectly via the profiler static walk; no dedicated module
- **Decisions**: [`ADR-0001`](../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md), [`ADR-0002`](../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
