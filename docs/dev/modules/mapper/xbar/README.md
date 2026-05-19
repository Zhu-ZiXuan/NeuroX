# Xbar Mapper Stack

This directory documents the current xbar-mapping stack.

Current structure:

- `XbarMapper` owns one tiler, one x slicer, and one w slicer
- `Tiler` handles geometry decomposition only
- `Slicer` handles value decomposition only
- `SimpleSlicer` uses direct digitise-then-group

See also:

- `../../../architecture/mapping.md`
- `../../../adr/ADR-0001-config-dispatch-and-owned-construction.md`
