# Python API

Two audiences share one package. An application calls the user API: units, file factories for units and primitives, and the measurement objects that read a run. A new device, circuit, macro, or unit is written against the extension SPI: the bases, mixins, and family doors below. The root `neurox` package is the aggregate user face; `neurox.api` itself exports no names, and its concrete modules hold the implementations lifted to that root. Each name's contract is stated in its owning docstring. The user API reference is rendered below; extension authors read the corresponding source interfaces.

## User API

- **Operators** — `neurox.architecture.unit` publishes `LinearUnit` and `Conv2dUnit`, exact-integer stand-ins for `torch.nn.functional.linear` and `torch.nn.functional.conv2d`. `program(weight, bias)` writes the static state; `linear(...)` / `conv2d(...)` run one call against it. What a unit computes is specified in [unit family](../reference/architecture/unit/family.md).
- **Implementations** — `neurox.architecture.unit.linear` and `neurox.architecture.unit.conv2d` each provide their operator base, config/policy family, and CIM and ideal implementations. Use `LinearUnit.from_config` or `Conv2dUnit.from_config` to select an implementation; `to_ideal()` constructs a fresh ideal unit.
- **File construction** — `neurox.linear_unit_from_file`, `neurox.conv2d_unit_from_file`, `neurox.cim_macro_from_file`, `neurox.iadc_from_file`, and `neurox.diff_vadc_from_file` are the family-specific file factories. Their shared loading contract and returned lifecycle state are rendered below from `neurox.api.factory`.
- **Configuration** — Each operator family exposes its own config and policy bases, with `from_file` and `from_preset` loading. The file contract is in [configuration](configuration.md).
- **Lifecycle** — a constructed unit is an `nn.Module`, so `.to(device)` and `.eval()` apply as usual. After migration, `neurox.fabricate(model)` resamples every NeuroX module registered anywhere below a user model; `unit.fabricate()` does the same for one unit subtree. What each call settles is in [physical state](../system_design/physical_state.md).
- **Temperature** — `neurox.set_temperature(model, T__K)` updates the NeuroX subtrees of an assembled model; `module.set_temperature(T__K)` updates one subtree, and `module.T__K` reads its current setting. Set the environment before explicitly triggering the fabrication or programming events that should use it.
- **Measurement** — `neurox.check_unique_binding` checks a model's physical-module ownership, and `neurox.stamp_names` enforces the same rule while naming an assembled model once; `neurox.Profiler` collects the dynamic-energy records a measured call emits, and `neurox.Reporter` turns one model plus one profiler into static and dynamic report rows.

The device, analog, digital, and macro models under `neurox.primitive` are what a unit composes below that surface. An application can construct a standalone family through its file factory. Within a composite, each child is built by its owner under the protocol in [construction](../system_design/construction.md). Extension code imports a device from its public module, for example `from neurox.primitive.device.mosfet import Nmos`; `neurox.primitive.device` exposes those modules rather than lifting their members. Their physics is specified in [Reference](../reference/README.md).

## Extension SPI

- **Module families** — inherit `neurox.common.module.ProfileModule` or `NonProfileModule`; both share the `ModuleBase` lifecycle. Use `ConfigBase` and `PolicyBase` for the family's configuration and policy.
- **Family dispatch** — inherit `neurox.common.registry_mixin.RegistryMixin` and register implementations with the family's `register_neurox_impl` decorator.
- **Serialization and validation** — use `neurox.common.serialize_mixin.SerializeMixin`, `neurox.common.validate_mixin.ValidateMixin`, and the functions in `neurox.common.serialize`.
- **Cross-module data** — inherit `neurox.common.module.SnapBase` for snapshots and `DcopBase` for DC operating points.
- **Recording** — inherit `neurox.common.recorder.RecorderBase` and `RecordBase`.
- **Operator interfaces** — `neurox.architecture.unit.UnitBase` and `UnitConfig` define shared operator metadata and local static costs. `LinearUnit` and `Conv2dUnit` define and register their operator implementations; `CimUnit` provides local CIM implementation helpers.
- **Digit encoding** — use `neurox.encoding.Transcoder.from_encoding` to select a transcoder.

System composition, physical-state lifetimes, and accounting responsibilities are described in [System design](../system_design/README.md).

The reference below is generated from the in-code docstrings by `mkdocstrings`.

::: neurox.api.factory
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical

::: neurox.api.function
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical

::: neurox.api.profiler
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical

::: neurox.api.reporter
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical

::: neurox.architecture.unit
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical
