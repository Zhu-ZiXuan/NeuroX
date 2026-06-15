# Macro

The macro is the simulated hardware architecture one level above a single [crossbar tile](../xbar/README.md): it organizes tiles into the unit that carries a full quantised-integer matrix multiply. Where a tile performs one primitive analog VMM over its native value domain, a macro decomposes a high-precision logical weight and activation across the tile layout, drives the per-tile reads, and aggregates the partial results into one integer output.

- [base](base.md) — the abstract macro contract: the integer-matmul protocol every family member satisfies and the value-domain / ADC surface it publishes.
- [xbar/](xbar/README.md) — the xbar-tile macro family: the abstract registry root, the three tiling modes (direct, inter-array slice, intra-array slice), and the lossless ideal twin.
