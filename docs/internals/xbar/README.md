# xbar — Implementation

How the crossbar layer is built. The spec is in [reference/xbar](../../reference/xbar/README.md); this side covers only what the code cannot tell you.

- [base](base.md) — `Xbar` family/registry, primitive shape contract, lifecycle, ideal twin.
- [cell](cell.md) — `XbarCell` family/registry, the no-PPA `nn.Module` device-owner, the abstract method set concrete cells implement.
- [solver](solver.md) — `Solver` family/registry, the stateless method-generic `solve_dc`, the nested solver, and the shared linear-algebra primitives.
- [_1t1r/](_1t1r/README.md) — the 1T1R implementation: chunked DC solve, offset orchestration.
- [readout/](readout/README.md) — the readout container implementation.
