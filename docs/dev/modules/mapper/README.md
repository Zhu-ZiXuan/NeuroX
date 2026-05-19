# Mapper Modules

This directory documents the mapper layer — the orchestrator that turns a high-precision logical weight / activation tensor into the xbar primitive contract. The mapper owns mapping semantics, tiling, and slicing.

## Files

- [`transcoder.md`](transcoder.md) — pure-algorithm signed-digit encoding policy (`true_form` / `complement` / `canonical`).
- [`xbar/`](docs/dev/modules/mapper/xbar/README.md) — the `XbarMapper` family plus the `slicer/` and `tiler/` sub-stacks.

See also:

- `docs/dev/architecture/mapping.md`
- `docs/dev/modules/xbar/README.md`
