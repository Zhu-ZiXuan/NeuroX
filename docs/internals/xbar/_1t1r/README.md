# 1T1R — Implementation

How the 1T1R topology is built. Spec: [reference/xbar/_1t1r](../../../reference/xbar/_1t1r/README.md).

- [cell](cell.md) — pluggable-cell family, access-node condensation, signed-conductance contract.
- [circuit_core](circuit_core.md) — chunked DC solve, energy accounting, wire buffers.
- [solver](solver.md) — solver framework, shared helpers, nested solver, benchmarks, calibration.
- [offset](offset.md) — `Offset1T1RXbar` orchestration (core + readout composition).
