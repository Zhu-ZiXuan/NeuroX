# Python API

The public Python surface stops at `neurox.architecture.unit`: a `QuantMatMul` is the unit a user wires into a model (see the [algorithm-engineer guide](../guides/algorithm_engineer/README.md)). Everything below it (`xbar`, `analog`, `device`, the unit-internal slicers, the `common.encoding` codec, ...) is internal and documented in [Reference](../reference/README.md) and [Internals](../internals/README.md).

The reference below is generated from the in-code docstrings by `mkdocstrings`.

::: neurox.architecture.unit
    options:
      show_root_heading: true
      show_source: false
      members_order: alphabetical
