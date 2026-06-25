# 1T1R array unit — Implementation

How the concrete 1T1R cell and the core array built for it are implemented. The spec is in [reference/xbar/_1t1r](../../../reference/xbar/_1t1r/README.md); this side covers only what the code cannot tell you.

- [cell](cell.md) — `XbarCell1T1R`: access-node condensation, fixed unrolled Newton, signed-conductance contract.
- [core](core.md) — `Core1T1R`: chunked DC solve, array-internal energy accounting, wire buffers, the `solve_array` entry that takes drivers in.
