# `neurox/common/mixin/registry.py`

## Current role

`RegistryMixin[KeyT, ImplT]` is a generic mixin that gives an abstract family root a `key → impl_class` registry plus a typed `_lookup_impl(key)` helper. Family roots that need polymorphic dispatch on either a config class or a string discriminator inherit it once with concrete type parameters.

It deliberately does **not** provide a single `from_*(...)` classmethod. Each family root declares its own factory (e.g. `from_config(...)`, `create(...)`) with the family's exact runtime-parameter signature, because different families need different explicit runtime parameters.

## Generic parameters

The mixin is `Generic[KeyT, ImplT]`:

- `KeyT` — the registry key's runtime type. Two common choices:
  - `KeyT = type[SomeConfig]` — config-class-keyed dispatch. Callers that hold a config instance `config` look it up via `cls._lookup_impl(type(config))`.
  - `KeyT = SomeLiteral` — string-discriminator dispatch (e.g. `Transcoder` keyed by the `Encoding` literal). Callers pass the literal directly: `cls._lookup_impl(encoding)`.
- `ImplT` — the family's impl base type.

`register_key(key)` and `_lookup_impl(key)` are typed through these type variables so a family's factory returns the family's impl base type rather than `Any`.

## Per-family registry lifecycle

`RegistryMixin.__init_subclass__` materialises a fresh `_impl_registry: dict[object, type]` on the **family root** the first time a subclass introduces the mixin. Sibling families therefore never share entries. Concrete impls inherit the family root's registry; their own `__init_subclass__` short-circuits because some ancestor already owns the dict.

Family roots must not redeclare `_impl_registry` by hand. The mixin encapsulates that mechanism entirely.

## `ClassVar` and `TypeVar` constraint

PEP 526 forbids `TypeVar` substitution inside `ClassVar`, so the mixin's on-disk type for `_impl_registry` is the broad `ClassVar[dict[object, type]]`. The parametric typing (`KeyT → type[ImplT]`) lives on the public-API surface via `Generic[KeyT, ImplT]`, and `_lookup_impl` casts the broad value type back to `type[ImplT]` at the return.

## Registering a concrete implementation

Concrete impls bind their key via the `register_key` decorator at class-definition time:

```python
# Config-class-keyed family
@SomeFamily.register_key(SomeConcreteConfig)
class SomeConcrete(SomeFamily): ...

# String-discriminator family
@SomeStringFamily.register_key("some_discriminator")
class SomeStringImpl(SomeStringFamily): ...
```

The decorator refuses to rebind a key already registered to a different impl class. Registration is driven by import side-effect; the family's package `__init__.py` must import every concrete impl module to populate the registry.

## Looking up an implementation

`_lookup_impl(key)` returns the impl class registered for `key`, or raises `TypeError` with a `known: ...` listing of registered keys. The error message renders type keys by `__name__` and other keys by `repr(...)`.

For config-class-keyed callers, the convention is to materialise the key from a config instance at the call site: `cls._lookup_impl(type(config))`. This keeps the `type()` step explicit and lets the mixin stay key-agnostic.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/adr/ADR-0001-config-dispatch-and-owned-construction.md`
