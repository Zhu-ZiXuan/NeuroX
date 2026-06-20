# 1T1R — Implementation

How the 1T1R topology is built. Spec: [reference/xbar/_1t1r](../../../reference/xbar/_1t1r/README.md). The DC solve itself is the topology-agnostic [solver](../solver.md) one level up.

- [cell](cell.md) — pluggable-cell family, access-node condensation, signed-conductance contract.
- [circuit_core](circuit_core.md) — chunked DC solve, energy accounting, wire buffers.
- [offset](offset.md) — `Offset1T1RXbar` orchestration (core + readout composition).
