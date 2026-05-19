# Common Utilities

This directory documents the `neurox.common` package — the lowest-level utility layer of NeuroX.

`neurox.common` carries no domain logic. It provides:

- config base classes and config I/O (`config.py`, `load_dump.py`)
- the polymorphic-family dispatch mixin (`config_dispatch.py`)
- shared physical constants (`physical_constant.py`)
- shared non-ideality kernels (`nonideality.py`)
- shared quantisation helpers (`quant.py`)

These primitives are intentionally small. They sit at the bottom of the import graph and depend on no other NeuroX subpackage.

See also:

- `../../architecture/config_and_construction.md`
- `../../adr/ADR-0001-config-dispatch-and-owned-construction.md`
