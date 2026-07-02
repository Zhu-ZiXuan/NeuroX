# Registry mixin

## Summary

`RegistryMixin[KeyT, ImplT]` gives an abstract family root a `key -> impl_class` registry plus a typed lookup helper, so a polymorphic family can dispatch on either a config class or a string discriminator. It provides exactly two public members — `register_key(key)` (a class-definition-time decorator) and `_lookup_impl(key)` — and deliberately ships no `from_*` factory: each family root declares its own factory (the canonical name is `from_config`) with the family's exact runtime-parameter signature. It is the private dispatch substrate behind keyed family construction and carries no physics.

## Design decisions

- **The mixin provides only `register_key` and `_lookup_impl`, never a `from_*` factory.** A generic mixin cannot know a family's runtime arguments, and a config object must not be able to instantiate an arbitrary family. So the mixin owns only the `key -> impl` mapping and the lookup; each family root declares its own factory (the canonical `from_config`, or a family-specific name) with explicit, family-specific runtime parameters.
- **A fresh registry is materialised per family root, not shared across siblings.** `__init_subclass__` creates a new `_impl_registry` dict on the family root the first time the mixin appears in a subtree; sibling families each get their own dict and never share entries. Concrete impls inherit the family root's dict and contribute no new one. This makes "which family am I registering into" a structural fact of the class tree rather than a hand-managed attribute.
- **Family roots must not declare `_impl_registry` by hand.** The materialisation mechanism is encapsulated entirely in the mixin; a hand-declared dict on a family root would either shadow or fight the auto-materialised one. Authors inherit the mixin and register; they never touch the storage.
- **Registration refuses to rebind an existing key.** `register_key` raises rather than silently overwrite an entry already bound to a different impl, so a duplicate key (e.g. two impls claiming the same config class or discriminator) is a load-time error, not a last-writer-wins surprise.
- **Registration is import-driven.** Binding happens as a side-effect of importing each concrete impl module, so a family's package `__init__.py` must import every impl that the family factory should be able to resolve; an un-imported impl is simply absent from the registry. There is no central registration table to maintain.
- **The on-disk storage type is broad; the parametric typing lives on the class.** PEP 526 forbids `TypeVar` substitution inside a `ClassVar`, so `_impl_registry` is annotated `ClassVar[dict[object, type]]` and the parametric `KeyT -> type[ImplT]` relation is carried by `Generic[KeyT, ImplT]` on the public surface, with `_lookup_impl` narrowing the broad value type back to `type[ImplT]` at its return.

## Contracts & invariants

- **Generic parameters.** `KeyT` is the registry key's runtime type; `ImplT` is the family's impl base type (bound to `RegistryMixin`). Both `register_key` and `_lookup_impl` are typed through these variables, so a family's factory gets back its impl base type rather than `Any`. Two key shapes are supported: `KeyT = type[SomeConfig]` for config-class-keyed dispatch, and `KeyT = SomeLiteral` for string-discriminator dispatch where the natural key is a value rather than a type.
- **`register_key(key)` returns a class decorator** applied at the impl's class-definition site. It binds `key` to the decorated class in the family root's registry and returns the class unchanged. If `key` is already bound to a different impl class it raises `TypeError`; rebinding a key to the same class is a no-op. The family root the decorator is invoked on (`@Family.register_key(...)`) selects which registry the binding lands in.
- **`_lookup_impl(key)` returns the impl class registered under `key`.** On a miss it raises `TypeError` whose message lists the known keys (type keys rendered by `__name__`, other keys by `repr(...)`), so a bad key surfaces the registered alternatives. The lookup performs no key coercion: config-class-keyed callers materialise the key at the call site as `cls._lookup_impl(type(config))`, keeping the mixin key-agnostic.
- **Per-family registry isolation.** A separate `_impl_registry` dict exists on each family root that introduces the mixin; entries registered on one family are invisible to siblings. Concrete impls share their family root's dict by inheritance and never own a separate registry.
- **Registry completeness depends on imports.** A `key` is only resolvable once its impl module has been imported. The family package `__init__.py` is responsible for importing every impl module; an un-imported impl is simply absent from the registry and yields the known-keys `TypeError` on lookup.

## Performance & resources

N/A — registration is a one-time import-side-effect and lookup is a single dict access, both off any per-VMM hot path.

## Gotchas

- **An un-imported impl is silently absent.** Because registration is an import side-effect, forgetting to import an impl module (typically a missing line in the family package `__init__.py`) makes its key unresolvable and surfaces only as a known-keys `TypeError` at lookup, not at definition time.
- **Do not hand-declare `_impl_registry` on a family root.** The mixin auto-materialises it; a manual declaration shadows the encapsulated mechanism and can break sibling isolation. Inherit the mixin and use `register_key`.
- **The mixin gives you no constructor.** Inheriting `RegistryMixin` does not provide a way to build an impl — the family root must define its own factory that calls `_lookup_impl` and supplies the family's runtime arguments.

## Known limitations

- **No automatic registration discovery.** Registry completeness is a hand-maintained import convention in each family's `__init__.py`; the mixin does not scan packages or verify that every impl has been registered.

---

- **Reference**: N/A — software dispatch substrate, no physics-bearing reference twin.
- **Implementation**: `neurox/common/mixin/registry.py`
- **Tests**: `tests/test_xbar_macro.py`, `tests/test_transcoder.py`
- **Decisions**: [ADR-0001](../../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)
