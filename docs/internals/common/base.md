# Base classes

`neurox/common/base.py` defines the three roots every NeuroX object descends from: `ConfigBase` (the root of every configuration dataclass), `PolicyBase` (the root of every policy dataclass), and `ModuleBase` (the root of every `nn.Module` — device, circuit, and orchestration unit). `ModuleBase` binds the typed `config` / `policy` pair, the construction-time `inst_shape`, the fabrication cascade, and the profile surface every module carries; which modules report static PPA is a per-class declaration.

## Design decisions

- **`ConfigBase` validates, `PolicyBase` does not.** A config both persists and validates, so `ConfigBase` composes `SerializeMixin` with `ValidateMixin` at the root and no config re-declares either. A policy is a frozen set of non-ideality switches whose legality is exhausted by their field types, so `PolicyBase` grants only `SerializeMixin` and deliberately omits `ValidateMixin` — this is the split between the two roots. Neither mixin hooks class creation, so their order at either root is unconstrained.
- **Every module is a profile host; static-PPA reporting is a per-class opt-out.** `ModuleBase` gives every module the emitter identity and the `area__um2` / `leakage__uW` surface, so the profile surface is uniform across the tree and only the reporting declaration varies. `reports_static_ppa` is a `ClassVar` defaulting true; a class whose silicon is already counted in an owner's budget declares it false, since self-reporting on top of that roll-up would double-count. TODO (domain author): the reason the profile surface is uniform across every module.
- **The owner constructs its children.** The config tree mirrors the ownership tree: when `A` owns `B`, `A`'s config carries `B`'s config as a field and `A.__init__` builds `B` directly, so ownership is readable from the config structure and construction from the owning class, with no external factory closure deciding which child class is instantiated. Polymorphic children are built through a per-family `from_config` factory that resolves the concrete class from the family's `RegistryMixin` key map.
- **Plain metaclass.** `FabricateMixin` is a plain class (no `ABCMeta`), so `ModuleBase`'s metaclass is plain `type`; abstractness is gated only where a subclass names `ABC` explicitly (the xbar cell, the CIM unit).

## Contracts & invariants

- **`inst_shape` is the fabrication multiplicity, committed once.** `inst_shape` is the per-instance multiplicity a module is built at — the same shape `fabricate()` samples its static state at — and it is fixed for the life of the module, so a subclass registers its buffers against it after delegating construction upward. `inst_count` is the parallel fabrication multiplicity behind one module, not a serial-op count. Both derive from init-fixed inputs, so both are read-only properties.
- **Construction-time dtype is a subsystem concern, not a base field.** `ModuleBase.__init__` takes no `dtype` / `T__K`; a subsystem base that owns physical buffers takes them in its own `__init__`, threads them top-down to the children it builds, and sizes its buffers with them. The numeric path reads the dtype off the buffer tensors, not a `self.dtype` lookup.
- **A config inherits `ConfigBase`; a policy inherits `PolicyBase`.** Each — directly or through a family base — is a frozen dataclass: the config gains the `{dict, file} × {read, write}` serialization surface from `SerializeMixin` plus the `validate*` helpers from `ValidateMixin`; the policy gains serialization only. The host wires when validation runs through its own `validate*` methods.

---

- **Reference**: N/A — software mechanism; per-instance area / leakage numbers are specified per subsystem under [reference](../../reference/README.md)
- **Implementation**: `neurox/common/base.py`
- **Tests**: TODO — no dedicated base test; serialization is exercised through `tests/common/test_serialize_mixin_use.py`, and `ModuleBase` indirectly through the profiler's static collection.
