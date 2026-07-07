# Xbar

How the crossbar layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — `Xbar` family/registry, primitive shape contract, lifecycle, ideal twin.
- [cell](cell.md) — `XbarCell` family/registry, the no-PPA `nn.Module` device-owner, the abstract method set concrete cells implement.
- [cell_1t1r](_1t1r/cell.md) — the concrete `XbarCell1T1R`: access-node condensation, fixed unrolled Newton, signed-conductance contract.
- [solver](solver.md) — `Solver` family/registry, the stateless method-generic `solve_dc`, the nested solver, and the shared linear-algebra primitives.
- [core/](_1t1r/README.md) — `Core1T1R`: the pure array's chunked DC solve, array-internal energy accounting, wire buffers.

Concrete scheme xbars that orchestrate drivers / reference / readout above a core live in their own per-scheme packages.
