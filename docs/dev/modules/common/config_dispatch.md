# `neurox/common/config_dispatch.py`

## Current role

`ConfigDispatchMixin[CfgT, ImplT]` is a generic mixin that gives an abstract family root a `config_type → impl_class` registry and a typed `_lookup_impl(cfg)` helper. Family roots that need polymorphic dispatch by concrete config type inherit it once with concrete type parameters.

It deliberately does **not** provide a single `from_config(...)` classmethod. Each family root declares its own `from_config(...)` with the family's exact runtime-parameter signature, because different families need different explicit runtime parameters.

## Generic parameters

The mixin is `Generic[CfgT, ImplT]`:

- `CfgT` — the family's base config type.
- `ImplT` — the family's impl base type.

`register_config(cfg_cls)` and `_lookup_impl(cfg)` are typed through these type variables so a family's `from_config(...)` returns its own impl base type rather than `Any`.

## Per-family registry lifecycle

`ConfigDispatchMixin.__init_subclass__` materialises a fresh `_config_registry: dict[type, type]` on the **family root** the first time a subclass introduces the mixin. Sibling families therefore never share entries. Concrete impls inherit the family root's registry; their own `__init_subclass__` short-circuits because some ancestor already owns the dict.

Family roots must not redeclare `_config_registry` by hand. The mixin encapsulates that mechanism entirely.

## `ClassVar` and `TypeVar` constraint

PEP 526 forbids `TypeVar` substitution inside `ClassVar`, so the mixin's on-disk type for `_config_registry` is the broad `ClassVar[dict[type, type]]`. The parametric typing (`type[CfgT] → type[ImplT]`) lives on the public-API surface via `Generic[CfgT, ImplT]`, and `_lookup_impl` casts the broad value type back to `type[ImplT]` at the return.

## Registering a concrete implementation

Concrete impls bind their config class via the `register_config` decorator at class-definition time:

```python
@SomeFamily.register_config(SomeConcreteConfig)
class SomeConcrete(SomeFamily):
    ...
```

The decorator refuses to rebind a config that is already registered to a different impl class.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/adr/ADR-0001-config-dispatch-and-owned-construction.md`
