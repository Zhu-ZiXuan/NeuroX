# Python API

The public Python surface stops at `neurox.architecture.unit`: a `LinearUnit` or `Conv2dUnit` is the unit a user wires into a model (see the [algorithm-engineer guide](../guides/algorithm_engineer/README.md)) — `UnitBase` roots the operator ABCs, the `CimUnit` family carries the engine-backed members (`LinearCimUnit`, `Conv2dCimUnit`) and the lossless references (`IdealLinearUnit`, `IdealConv2dUnit`), all built through `CimUnit.from_config`. Everything below it (`xbar`, `analog`, `device`, the unit-internal slicers, the `common.encoding` codec, ...) is internal and documented in [Reference](../reference/README.md) and [Internals](../internals/README.md).

The reference below is generated from the in-code docstrings by `mkdocstrings`.

::: neurox.architecture.unit
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical
