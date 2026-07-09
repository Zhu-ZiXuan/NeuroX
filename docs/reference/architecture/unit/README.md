# Unit

The unit is the simulated hardware architecture one level above a single [crossbar tile](../../primitive/xbar/README.md): it composes tiles into the entity that carries a full quantised-integer matrix multiply. Where a tile performs one primitive analog VMM over its native value domain, a unit places a high-precision matmul on the tile layout along two orthogonal axes — it decomposes each value into positional slices ($S_w$, $S_a$, the precision-slicing axis), tiles the matmul across the physical grid ($T_r$, $T_c$, the matrix-tiling axis), drives the per-tile reads, and aggregates the partial results back into one integer output by a radix-weighted shift-add.

- [base](base.md) — the abstract unit contract: the integer-matmul protocol every family member satisfies, the two axes (precision slicing $S_w$/$S_a$, matrix tiling $T_r$/$T_c$) and their decompose $\leftrightarrow$ aggregate dual, and the value-domain / ADC surface it publishes.
- [cim/](cim/README.md) — the xbar-tile unit family: the three tiling modes (direct, inter-array slice, intra-array slice) and the lossless ideal twin.
