# 1T1R Topology

The 1T1R crossbar: one RRAM device plus one access transistor per cell. A shared physical array and the operating xbars that add a readout to make it compute. The array's DC operating-point solve is the topology-agnostic [solver](../solver.md) one level up.

- [cell](cell.md) — the single cell as a two-terminal element to the solver: condensed access node, signed branch conductances, device-capacitor energy.
- [circuit_core](circuit_core.md) — the shared physical array: model, governing equations, energy.
- [offset](offset.md) — the offset-coded operating xbar (circuit_core + readout).

A future differential 1T1R xbar is a sibling here: a different operating xbar reusing the same [circuit_core](circuit_core.md).
