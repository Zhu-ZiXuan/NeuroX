# Xbar

How the crossbar array layer is built. This side covers only what the code cannot tell you.

- [cell](cell/README.md) — `XbarCell` family/registry, the no-PPA `nn.Module` device-owner, the abstract method set concrete cells implement.
- [cell_1t1r](cell/_1t1r/cell.md) — the concrete `XbarCell1T1R`: access-node condensation, fixed unrolled Newton, signed-conductance contract.
- [array](array/_1t1r/array.md) — `XbarArray1T1R`: the pure array's chunked DC solve, array-internal energy accounting, wire buffers.
- [solver](solver.md) — `Solver` family/registry, the stateless method-generic `solve_dc`, the nested solver, and the shared linear-algebra primitives.

The tile-level `CimMacro` family/registry, primitive shape contract, lifecycle, and ideal twin live in [macro/cim/base](../macro/cim/base.md). Concrete scheme xbars that orchestrate drivers / reference / readout above an array live in their own per-scheme packages.
