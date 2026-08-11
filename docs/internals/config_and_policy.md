# Config and policy

Every module is configured by two parallel objects passed together at construction as `Module(config=..., policy=...)`. Its `config` is the immutable physical, design, and specification reality it is built from — a datum fixed by the process, by the design, or by measurement, reused unchanged across runs. Its `policy` is the per-run set of runtime parameters applied to that reality — chiefly, but not only, which non-idealities to model — free to differ from one run to the next. One fixed config is thus exercised under many runtime stances. The Source of each config value and the config layer that fixes it are classified in [module_parameter](../conventions/module_parameter.md).

Every module class defines config and policy types, retaining an explicit empty class when necessary. Names normally follow `XxxConfig` and `XxxPolicy`; a family may deliberately share one or both types. Construction always receives both objects rather than replacing an empty object with `None`, a default, or an omitted argument.

## Explicit over implicit

- **No defaults.** The core library supplies no default value for any config field or constructor parameter. Every value is set explicitly — by a config file, or by the owning holder that constructs the module — so an unset value is an error, not a silent fallback.
- **`None` is not a default.** `None`, likewise, never encodes "disabled" or "an implicit default": switching a non-ideality off is a policy decision, and an absent quantity is simply absent, not a hidden zero.
- **Field types are module-local.** The concrete type each field takes varies from module to module, and is stated by that module's own documentation rather than fixed here.
- **Exceptions are documented locally.** A few genuine exceptions to the no-default rule exist; each is called out by the documentation of the module that owns it.

## Noise parameters and toggles

- **`config` carries parameters.** A `config` carries the noise parameters — the magnitudes taken from device physics or from measurement.
- **`policy` carries toggles.** Runtime parameters for those sources are per-source toggles — for each modelled non-ideality, the decision to apply or skip it on this run.
- **Not one-to-one.** Parameters and toggles are not in one-to-one correspondence. Some sources scale a relative noise off a quantity that is already present and so hold no parameter of their own, carrying only a toggle. In the other direction, a config field that is not a noise magnitude — a structural construction choice, or a numerical hyperparameter — carries no toggle at all.
- **Decision lives on `policy`.** Either way, the apply / skip decision lives on the policy and never on the config: a config field states a value, never whether that value is used. Folding that decision onto the config as an `enable_*` flag would conflate a source's magnitude with the decision to model it, and would bind that decision to the immutable design.

## Owner constructs its children

The config tree mirrors the ownership tree.

- **Config carries the child.** When `A` owns `B`, `A`'s config carries `B`'s config as a field, so ownership is readable from the config structure, and `A.__init__` builds `B` directly — no external factory closure decides which child class is instantiated.
- **Polymorphic children dispatch through the family.** A child chosen from an implementation family is built through the family's `from_config`. Each leaf uses `register_neurox_module(config_type=..., policy_type=...)`; lookup receives the config and policy objects separately, while `RegistryMixin` alone forms the internal type-pair key. A mismatched pair therefore fails at dispatch rather than inside a leaf constructor.
- **A child config is construction material only.** The owner hands the child config to the family's `from_config` and never reads a field back out of it afterwards. What the owner needs at run time it reads off the constructed child's declared surface — a method or a property the child publishes — and what the owner computes for itself belongs on the owner's own config. Reaching into a child's config would put the child's own interpretation of its fields in a second place, where it drifts from the child silently.
- **Field placement follows variability.** A parameter that is fixed across instances and across applications — a characteristic of the module itself — is a field on that module's own config; a parameter that changes with the instance, the layout, or the deployment is a field on the holder's config, since it is the holder's placement that varies it. The parameter layers this criterion sorts into are defined in [module_parameter](../conventions/module_parameter.md).

- **Instance multiplicity is never empty hardware.** `inst_shape=()` means one instance with no replication axes. Every explicit extent must be positive; a shape containing zero or a negative extent is rejected by `ModuleBase`, so `inst_count` is always positive.

## Policy mirrors config

A `policy` mirrors its `config` in structure.

- **Leaf module.** A leaf module's policy is a flat set of runtime parameters.
- **Composite.** A composite's policy nests its children's policies exactly as the composite's config nests their configs.
- **Implementation family.** Where a config selects one implementation from a family, the policy takes the same shape — an abstract marker for the family, with one concrete policy per implementation — and the caller passes the concrete policy that matches the chosen config.
