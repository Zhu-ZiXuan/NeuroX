# Construction

A run turns configuration files into a live module tree: every module is built by its owner from an immutable `config` and a per-run `policy`, and the types of that pair select one implementation out of a family. The protocol spans the serialization layer, the registry mixin, the module roots, and every owner's `__init__`; a change at one of those ends holds only if the others honor the same contract.

## The two objects

A module is always constructed as `Module(config=..., policy=...)`.

- `config` is the immutable physical, design, and specification reality the module is built from — a datum fixed by the process, by the design, or by measurement, reused unchanged across runs. Each value's Source and layer are classified in [module parameter](../conventions/module_parameter.md).
- `policy` is the runtime stance applied to that reality on one run — chiefly, but not only, which non-idealities to model. One fixed config is exercised under many stances.

Every module instance receives both objects. A module class may declare its role types, inherit them from a parent, or share them with siblings whose configuration and runtime stance are genuinely the same. Names normally follow `XxxConfig` and `XxxPolicy` as a navigation convention, not as a dispatch rule. An empty type is retained only when it carries a semantic or dispatch identity; construction never replaces either object with `None`, a default, or an omitted argument.

A noise source's magnitude and the decision to model it are split across the pair: the magnitude, from device physics or from measurement, is a config field; the apply-or-skip decision is a policy field. The two sets are not in one-to-one correspondence — a source that scales a relative noise off a quantity already present holds no magnitude of its own and carries only a toggle, and a config field that is a structural choice or a numerical hyperparameter carries no toggle at all. A config field states a value, never whether that value is used: an `enable_*` flag living on a config would bind a per-run decision to the immutable design. For the same reason `None` encodes neither "disabled" nor an implicit default — an absent quantity is absent, and switching a source off is a policy decision.

A policy mirrors config ownership in structure. A leaf's policy is a flat set of runtime parameters, and a composite's policy nests its children's policies exactly as its config nests their configs. At a registry slot the concrete type pair selects the implementation; either side of that pair may be shared when the implementation distinction lives entirely in the other type.

## Owner constructs its children

The config tree mirrors the ownership tree. When `A` owns `B`, `A`'s config carries `B`'s config as a field, so ownership is readable off the config structure, and `A.__init__` builds `B` directly — no external factory closure decides which child class is instantiated. A child with no alternatives is constructed from the pair its owner already holds; a child drawn from an implementation family is built through the family's `from_config`.

Beside the pair, the owner passes what its own placement fixes — instance multiplicity, logical extents, dtype, operating temperature — as keyword arguments. Which of the two configs a parameter belongs on follows the variability criterion in [module parameter](../conventions/module_parameter.md#config-layering).

A child config is construction material only. The owner hands it to the child and never reads a field back out of it: what the owner needs at run time it reads off the constructed child's declared surface — a property or a method the child publishes — and what the owner computes for itself belongs on the owner's own config. Reaching into a child's config would place the child's interpretation of its own fields in a second location, where it drifts from the child silently.

## Registry dispatch

A family base mixes in `RegistryMixin` and owns exactly one table for the whole family, keyed by the pair of concrete types `(config type, policy type)`; each concrete implementation binds itself to one pair through `register_neurox_module`. Lookup receives the config and policy objects and reads nothing but their types.

Keying on the pair rather than on the config alone makes the config-policy correspondence a dispatch condition in its own right: a policy that does not match the chosen config selects no implementation, so the mismatch fails at the family door rather than surfacing as a missing field deep inside a leaf constructor.

The table is private to the family. Callers reach it only through the public `from_config` classmethod the family base publishes, which resolves the implementation and forwards the owner's construction arguments unchanged. A family with a single implementation publishes the same door, so an owner's construction call does not change when that family gains a second one.

## Configuration files

Config and policy load as a paired, parallel file set — the design parameters in one file, the run's policy in another — each resolved and coerced into its dataclass tree independently. Nothing in the file format binds one file to the other: the pair is formed by the caller and checked at dispatch. File formats, reserved directives, and field coercion are documented in [configuration](../api/configuration.md); the file set of a calibration campaign and the provenance rules its values carry are documented in [campaigns](../validation/campaigns.md).

A `_neurox_class` discriminator names the concrete class for a polymorphic slot, and the name resolves only within the receiver's own subclass graph, so the field being filled bounds what a file can construct. A table that pulls in a shared fragment or a bundled preset may not also declare that key: the referenced file is the sole authority for the class it describes, and the referencing site may only override values it defines. Class identity therefore travels with the parameter set that defines it, and a shared preset cannot be re-pointed from a call site at a class that never declared the fields the preset supplies.

## Import graph

Both selection mechanisms are populated by import side effects, and each sees only what has been imported: `register_neurox_module` binds its pair when the implementing module executes, and a `_neurox_class` name resolves against the subclass graph as it stands at load time. An implementation whose module was never imported is invisible to both, and the failure reads as an unknown class or an unknown pair rather than as a missing import.

Dispatch completeness is therefore a property of the import graph rather than of any single family. Every package imports the subpackages and modules beneath it, so `import neurox` loads the whole implementation tree, including the bundled works whose schemes register into core family bases. Callers and tools perform no registration-only import, and no family maintains a list of its own implementations. The direction of those imports is a separate rule, machine-checked by `tests/rules/test_import.py`.
