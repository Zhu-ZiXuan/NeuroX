# Python API

Two audiences share one package. An application calls the user API: units, the file loaders that build them, and the measurement objects that read a run. A new device, circuit, macro, or unit is written against the extension SPI: the bases, mixins, and family doors below. Each name's own contract is stated in its docstring — rendered below for the unit surface and in [Common API](common.md) for the shared kernel — so this page says only which surface a name belongs to.

## User API

- **Operators** — `neurox.architecture.unit` publishes `LinearUnit` and `Conv2dUnit`, exact-integer stand-ins for `torch.nn.functional.linear` and `torch.nn.functional.conv2d`. `program(weight, bias)` writes the static state; `linear(...)` / `conv2d(...)` run one call against it. What a unit computes is specified in [unit family](../reference/architecture/unit/family.md).
- **Implementations** — `neurox.architecture.unit.cim` holds the engine-backed units `LinearCimUnit` and `Conv2dCimUnit`; `neurox.architecture.unit.ideal` holds the lossless references `IdealLinearUnit` and `IdealConv2dUnit`. All four are built through one door, `CimUnit.from_config`, which selects the implementation from the types of the config and policy pair it receives.
- **Configuration** — `CimUnitConfig.from_file` and `CimUnitPolicy.from_file` read that pair out of a file set, and `from_preset` reads one bundled fragment directly. The file contract is in [configuration](configuration.md).
- **Lifecycle** — a constructed unit is an `nn.Module`, so `.to(device)` and `.eval()` apply as usual. After migration, `neurox.fabricate(model)` resamples every NeuroX module registered anywhere below a user model; `unit.fabricate()` does the same for one unit subtree. What each call settles is in [physical state](../system_design/physical_state.md).
- **Measurement** — `neurox.stamp_names` names an assembled model once, `neurox.Profiler` collects the dynamic-energy records a measured call emits, and `neurox.Reporter` turns one model plus one profiler into static and dynamic report rows. `neurox.common.neurox_roots` finds the outermost NeuroX modules a mixed tree holds, so a script reaches them without hard-coding its own layer layout.

The device, analog, digital, and macro models under `neurox.primitive` are what a unit composes below that surface. An application configures them through the file set rather than constructing them: each is built by its owner, under the protocol in [construction](../system_design/construction.md). Their physics is specified in [Reference](../reference/README.md).

## Extension SPI

- **Module root** — `neurox.common.ModuleBase`, with the immutable value objects `ConfigBase` and `PolicyBase` it is constructed from. It supplies the config and policy properties, the instance multiplicity, the local fabrication-variation hook, and the static PPA hooks a profile target answers; the public `fabricate` entry point owns traversal.
- **Family dispatch** — `neurox.common.RegistryMixin`. A family base mixes it in and owns one table keyed by the concrete `(config type, policy type)` pair; each implementation binds itself with `register_neurox_module`, and the base publishes a `from_config` door callers use instead of the table.
- **Serialization and validation** — `neurox.common.SerializeMixin` gives a config or policy dataclass its mapping, file, and preset constructors; `neurox.common.ValidateMixin` supplies the predicates a `validate` method raises through. The function-level surface lives in `neurox.common.serialize`.
- **Cross-module data** — `neurox.common.SnapBase`, `DcopBase`, and `RecordBase` identify sampled state, DC operating points, and recorder rows. They use `TensorDataClassBase` for shared dataclass semantics; `SnapBase` also includes the per-field shape operations of `TensorGroupMixin`, built on `walk_tensor_fields`.
- **Recording** — `neurox.common.RecorderBase` and `RecordBase` are the side-channel collection family the profiler and the probers specialize.
- **Operator interfaces** — `neurox.architecture.unit.UnitBase` and its `LinearUnit` / `Conv2dUnit` leaves define the lowering seams a new operator fills; `CimUnit` is the registry-dispatched base a new CIM unit registers into.
- **Matmul planning** — `neurox.architecture.unit` also exports the substrate-independent placement plans `MatmulPlacementPlan`, `InputActivationPlan`, and `BlockSlotRouting` with their `make_*` builders, which turn one logical matrix multiply into the block geometry any fixed-capacity substrate maps it onto ([placement](../reference/architecture/unit/cim/engine/placement.md)).
- **Digit encoding** — `neurox.common.encoding.Transcoder`, with `Encoding` naming the shipped algorithms and `create_transcoder` building one.

A contract that no single one of these bases owns — the construction protocol, the state lifecycle, the PPA accounting axes, the compile boundary — is in [System design](../system_design/README.md).

The reference below is generated from the in-code docstrings by `mkdocstrings`.

::: neurox.architecture.unit
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical
