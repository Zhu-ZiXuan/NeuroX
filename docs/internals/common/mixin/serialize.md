# Serialize mixin

`SerializeMixin` is the object-oriented front onto the [serialize](../serialize/README.md) machinery: the `{dict, file} x {read, write}` surface plus `from_preset` for a frozen dataclass.

## Design decisions

- **A pure, thin front.** `SerializeMixin` defines no free function of its own: every method body is a one-line forward into the `neurox.common.serialize` package, so a host reaches the surface through `self` / `cls`. The mixin owns no field, runs no validation, and holds no state. Validation is a separate concern a host composes elsewhere, so inheriting this mixin persists a host but does not validate it.
- **`section` is a per-call argument, not a property.** `section` selects one top-level table of a multi-config file; it depends on the file being read or written, not on the type, so it is a per-call file-layout argument on every `{dict, file} x {read, write}` method, never a property or field of the host.
- **Receiver-bounded constructor.** `from_dict` / `from_file` invoked on a `Root` resolve a `_neurox_class` name only within `Root`'s own subclass subtree: the returned instance is `Root` or one of its subclasses, never anything else. A discriminator that names a `ConcreteScheme` living outside `Root`'s subtree is an error, not a lookup that reaches across the class graph. A receiver therefore bounds what it can construct — a call on a narrow `Family` cannot be tricked by file data into building an unrelated type. This inherits directly from the machinery's [receiver-bounded constructor and abstract-base rule](../serialize/README.md).
- **`from_preset` is a thin convenience over `from_file`.** It parses a `"family/file:section"` reference into a path under `neurox/presets/` and a section name, then delegates to `from_file`, inheriting the same receiver-bounded and abstract-base rules. The inline and dotted-key preset idioms — `field = { _neurox_use_preset = "family/file:section" }` or `field._neurox_use_preset = "family/file:section"` embedded inside a user config — reach the identical bundled preset through directive resolution rather than `from_preset`; the two entry points (a top-level call on the target type, and a directive nested inside another file) resolve the same `"<path-under-presets-root>:<section>"` grammar.

## Composition

Because `SerializeMixin` carries no construction or class-creation hook, it constrains no method-resolution order, and a host may compose it in any position.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/serialize.py`
- **Tests**: `tests/common/test_serialize_mixin_use.py`, `tests/common/test_load_dump_use.py`, `tests/common/test_preset_authority.py`
</content>
