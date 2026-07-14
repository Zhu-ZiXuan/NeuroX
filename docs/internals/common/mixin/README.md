# Mixin

The reusable mixin classes composed into NeuroX circuit modules and config dataclasses.

- [fabricate](fabricate.md) — `FabricateMixin`, the pre-order `fabricate()` cascade for manufacturing-variation sampling.
- [profile](profile.md) — `ProfileMixin`, the per-module emitter of the PPA side channel.
- [registry](registry.md) — `RegistryMixin`, the implementation registry and typed lookup for polymorphic families.
- [serialize](serialize.md) — `SerializeMixin`, the `{dict, file} x {read, write}` surface plus `from_preset` for frozen dataclasses, fronting the [serialize/](../serialize/README.md) machinery.
- [validate](validate.md) — `ValidateMixin`, the static check helpers for a host's `validate` methods.
