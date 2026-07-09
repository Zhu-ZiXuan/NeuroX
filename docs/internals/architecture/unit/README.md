# Unit

How the unit layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the cross-unit `QuantMatMul` Protocol, the not-a-circuit (zero-PPA) decision, and the `[Sa, Sw, Tc, Tr]` layout convention shared across modes.
- [cim/](cim/README.md) — the `CimUnit` registry family: registry dispatch, the tile-build helper, the chunk-and-pad primitive, and the per-mode organize/aggregate shape pipelines.
