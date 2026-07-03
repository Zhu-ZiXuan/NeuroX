# Registry mixin

## Summary

`RegistryMixin[KeyT, ImplT]` grants a family root a `key -> impl_class` registry plus a typed lookup, so a polymorphic family can dispatch construction on either a config class or a string discriminator. It owns only the mapping and its access — the `register_key` decorator that binds an impl and the `_lookup_impl` helper that resolves one. It deliberately ships no `from_*` factory and no constructor: building an impl, and fixing the family's runtime-parameter signature, stays with the family root. It is the dispatch substrate behind keyed family construction and carries no physics.

## Design decisions

- **The mixin owns only the mapping and the lookup, never a `from_*` factory.** A generic mixin cannot know a family's runtime arguments, and a config object must not be able to instantiate an arbitrary family. So the mixin holds the `key -> impl` map plus `register_key` and `_lookup_impl`, and each family root declares its own factory — canonically `from_config` — with the family's explicit runtime parameters.
- **A fresh registry is materialised per family root, not shared across siblings.** Treating "which family am I registering into" as a structural fact of the class tree, rather than a hand-managed attribute, keeps sibling families from leaking entries into one another and frees an impl from carrying a family pointer. The materialisation mechanism is encapsulated in the mixin, which is why a family root must never hand-declare the storage.
- **Registration refuses to rebind an existing key.** A duplicate key — two impls claiming the same config class or discriminator — is a load-time `TypeError`, not a last-writer-wins surprise; rebinding a key to the same class stays a no-op so a re-imported module is harmless.
- **Registration is import-driven, with no package scan.** Binding is a side-effect of importing an impl module, so a family keeps no central registration table; the cost is that completeness depends on the family package importing every impl, and an un-imported impl is simply absent.
- **The stored value type is broad; the parametric typing lives on the class.** PEP 526 forbids `TypeVar` substitution inside a `ClassVar`, so `_impl_registry` is annotated `ClassVar[dict[object, type]]` while `Generic[KeyT, ImplT]` carries the `KeyT -> type[ImplT]` relation on the public surface and `_lookup_impl` narrows the broad value back to `type[ImplT]` at its return.

## Composition

`RegistryMixin` participates through the `__init_subclass__` class-creation hook and cooperatively forwards `super().__init_subclass__(**kwargs)`, so it composes with any other mixin that also hooks subclass creation regardless of MRO order.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/registry.py`
- **Tests**: `tests/test_xbar_macro.py`, `tests/test_transcoder.py`
- **Decisions**: [ADR-0001](../../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)
