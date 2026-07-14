# Base classes

## Summary

`neurox/common/base.py` defines the three roots every NeuroX object descends from: `ConfigBase` (the root of every configuration dataclass), `PolicyBase` (the root of every policy dataclass), and `ModuleBase` (the root of every `nn.Module` — device, circuit, and orchestration unit). `ModuleBase` binds the typed `config` / `policy` pair, the construction-time `inst_shape`, and the fabrication cascade; it carries no PPA surface, which sized subclasses add through `ProfileMixin`.

## Design decisions

- **`ConfigBase` validates, `PolicyBase` does not.** A config both persists and validates, so `ConfigBase` composes `SerializeMixin` with `ValidateMixin` at the root and no config re-declares either. A policy is a frozen set of non-ideality switches whose legality is exhausted by their field types, so `PolicyBase` grants only `SerializeMixin` and deliberately omits `ValidateMixin` — this is the split between the two roots.
- **`ModuleBase` carries no static PPA.** The base composes `FabricateMixin` + `nn.Module` only; it owns no `area__um2` / `leakage__uW` surface. A bare `ModuleBase` leaf — a device, the current mirror / mux, an xbar cell — owns no silicon to report, so PPA is not on the base: a sized module adds `ProfileMixin` (which owns the static-PPA aggregation) and sets the bare per-instance data it aggregates. Rejected — a fixed PPA-carrying base every module inherits: it would force devices, cells, and orchestration nodes that own no reportable silicon to carry, and zero out, a surface they do not have.
- **Typed `config` / `policy` pair.** `ModuleBase` is `Generic[ConfigT, PolicyT]`, so a subclass declared `Leaf(ModuleBase[LeafConfig, LeafPolicy])` narrows `self.config` / `self.policy` without a per-leaf annotation; a parameterized `Generic` can still lose that precision through `nn.Module.__getattr__`, which the leaf restores with class-level `config: LeafConfig` / `policy: LeafPolicy` forward declarations alongside its buffer declarations.
- **`name` is optional and owner-supplied.** `__init__` takes `name: str = ""`; a leaf built as a hierarchy member receives its dotted `name` from the owner, while a device or cell — never a direct profiler emitter — takes the default. `ModuleBase` binds `self._neurox_name` directly (the name `ProfileMixin.qualified_name` reads) rather than cooperatively through `ProfileMixin.__init__`, because `nn.Module.__init__` is not cooperative.
- **The owner constructs its children.** The config tree mirrors the ownership tree: when `A` owns `B`, `A`'s config carries `B`'s config as a field and `A.__init__` builds `B` directly, so ownership is readable from the config structure and construction from the owning class, with no external factory closure deciding which child class is instantiated. Polymorphic children are built through a per-family `from_config` factory ([registry](mixin/registry.md)).
- **Plain metaclass.** `FabricateMixin` is a plain class (no `ABCMeta`), so `ModuleBase`'s metaclass is plain `type`; abstractness is gated only where a subclass names `ABC` explicitly (the xbar cell, the CIM unit).

## Contracts & invariants

- **Construction signature.** `__init__(self, *, config: ConfigT, policy: PolicyT, name: str = "", inst_shape: tuple[int, ...])` is keyword-only. It runs `nn.Module.__init__`, binds `config` / `policy` / `_neurox_name`, and records `inst_shape` — the per-instance fabrication multiplicity, committed once for the life of the module. A subclass `__init__` calls `super().__init__(...)` first, then registers its buffers and extra fields. `FabricateMixin` contributes no initializer and passes through the chain.
- **`inst_shape` / `inst_count` properties.** `inst_shape` returns the construction-time per-instance multiplicity — the same shape the fabrication state is sampled at; `inst_count` is its `math.prod`, the parallel fabrication multiplicity behind one module, not a serial-op count. Both are functions of init-fixed inputs, so both are properties with no setter.
- **Construction-time dtype is a subsystem concern, not a base field.** `ModuleBase.__init__` takes no `dtype` / `T__K`; a subsystem base that owns physical buffers takes them in its own `__init__`, threads them top-down to the children it builds, and sizes its buffers with them. The numeric path reads the dtype off the buffer tensors, not a `self.dtype` lookup.
- **A config inherits `ConfigBase`; a policy inherits `PolicyBase`.** Each — directly or through a family base — is a frozen dataclass: the config gains the `{dict, file} × {read, write}` serialization surface from `SerializeMixin` plus the `validate*` helpers from `ValidateMixin`; the policy gains serialization only. The host wires when validation runs through its own `validate*` methods.

## Composition

- **`ConfigBase(SerializeMixin, ValidateMixin)` / `PolicyBase(SerializeMixin)`.** Neither mixin hooks class creation, so their order is unconstrained.
- **`ModuleBase(FabricateMixin, nn.Module, Generic[ConfigT, PolicyT])`.** The composition fixes the MRO and the `__init__` chain every subclass inherits; the metaclass stays plain `type`.

---

- **Reference**: N/A — software mechanism; per-instance area / leakage numbers are specified per subsystem under [reference](../../reference/README.md)
- **Implementation**: `neurox/common/base.py`
- **Tests**: TODO — no dedicated base test; serialization is exercised through `tests/common/test_serialize_mixin_use.py`, and `ModuleBase` indirectly through the profiler's static collection.
