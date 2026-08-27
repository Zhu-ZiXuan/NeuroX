# Common API

`neurox.common` is the shared kernel every NeuroX module is built on, in six groups:

- **Module root** — `ModuleBase`, with the immutable `ConfigBase` and `PolicyBase` value objects it is constructed from, plus `fabricate`, which realizes every NeuroX module registered below an arbitrary `nn.Module` root.
- **Serialization** — `SerializeMixin` and the `serialize` subpackage, which turn a config or policy file into a dataclass tree, plus `ValidateMixin` for the checks that tree runs on construction.
- **Family dispatch** — `RegistryMixin`, one table per implementation family keyed by the concrete config and policy types.
- **Recording** — `RecorderBase` and `RecordBase`, the side-channel collection family; `Profiler` and `EnergyRecord`, its dynamic-energy specialization; and `Reporter` with the `StaticEntry`, `DynamicEntry`, and `StaticMetrics` rows it renders.
- **Cross-module data** — `SnapBase`, `DcopBase`, and `RecordBase` name sampled state, DC operating points, and recorder rows. They share the construction semantics of `TensorDataClassBase`; `SnapBase` also supplies the per-field shape operations of `TensorGroupMixin`, built on `walk_tensor_fields`.
- **Module naming** — `check_unique_neurox_bindings` checks that each physical module occupies one path, while `stamp_names` enforces that rule and names every NeuroX module below an arbitrary `nn.Module`; `ModuleBase.stamp_names` owns the naming operation for one NeuroX subtree.

The `encoding` subpackage sits alongside them, holding the fixed-length signed-digit transcoders that convert an integer to a positional digit representation and back.

The reference below is generated from the in-code docstrings by `mkdocstrings`.

::: neurox.common
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical
