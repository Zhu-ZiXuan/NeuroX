# Xbar

How the crossbar array layer is built. This side covers only what the code cannot tell you.

- [cell](cell/README.md) — the `XbarCell` family, the device-owning module, the abstract method set concrete cells implement.
- [cell_1t1r](cell/_1t1r/cell.md) — the abstract `XbarCell1t1r` base: shared node-cap substrate, single family DCOP, shared grounded-cap energy formula.
- [cell_1t1r_detail](cell/_1t1r/cell_detail.md) — `XbarCell1t1rDetail`: nonlinear device children, access-node condensation, fixed unrolled Newton.
- [cell_1t1r_linear](cell/_1t1r/cell_linear.md) — `XbarCell1t1rLinear`: table-driven closed-form branch, program-time gather, empty policy.
- [array](array/_1t1r/array.md) — `XbarArray1t1r`: the pure array's chunked DC solve, array-internal energy accounting, wire buffers.
- [solver](solver.md) — `Solver` family/registry, the stateless method-generic `solve_dc`, the nested solver, and the shared numerical solver helpers.

The tile-level `CimMacro` family/registry, primitive shape contract, lifecycle, and ideal twin live in [macro/cim/base](../macro/cim/base.md). Concrete scheme xbars that orchestrate drivers / reference / readout above an array live in their own per-scheme modules.
