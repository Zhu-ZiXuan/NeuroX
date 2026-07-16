# CimUnit family

The unit family that organizes [crossbar tiles](../../../primitive/xbar/README.md): it decomposes each value into positional slices (the precision-slicing axis $S_w$, $S_a$) and lays them onto the physical tile grid (the matrix-tiling axis $T_r$, $T_c$), then aggregates the per-tile reads back into one integer matmul by a radix-weighted shift-add. The modes differ only in how the precision-slicing axis maps onto the tiling layout.

- [direct](direct.md) — the no-slice corner $S_w = S_a = 1$: weights and activations map straight onto one tile's native value range.
- [inter_array_slice](inter_array_slice.md) — weight slices distributed across separate tile planes, recombined by cross-tile shift-add.
- [intra_array_slice](intra_array_slice.md) — weight slices gathered into adjacent columns of one tile, recombined by intra-tile shift-add.
- [ideal](ideal.md) — the lossless integer-matmul twin: no tile, the value-domain reference any physical mode is compared against.

The abstract contract these share — the tiling scheme, the value domain, and the ADC surface — is in [unit/family](../family.md).
