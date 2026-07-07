# Macro

How the macro layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the cross-macro `NeuroxMacroQuantMatMul` Protocol, the not-a-circuit (zero-PPA) decision, and the `[Sa, Sw, Tc, Tr]` layout convention shared across modes.
- [xbar/](xbar/README.md) — the `XbarMacro` registry family: registry dispatch, the xbar-build helper, the chunk-and-pad primitive, and the per-mode organize/aggregate shape pipelines.
