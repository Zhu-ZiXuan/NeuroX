# XbarMacro Family

The macro family that organizes [crossbar tiles](../../xbar/README.md): it decomposes a logical weight and activation across tiles via a slice-organize-tile scheme and aggregates the per-tile reads back into one integer matmul. The modes differ only in how the per-weight slice axis maps onto the physical tile layout.

- [direct](direct.md) — the no-slice mode: weights and activations map straight onto one tile's native value range (`Sw = Sa = 1`).
- [inter_array_slice](inter_array_slice.md) — weight slices distributed across separate tile planes, recombined by cross-tile shift-add.
- [intra_array_slice](intra_array_slice.md) — weight slices gathered into adjacent columns of one tile, recombined by intra-tile shift-add.
- [ideal](ideal.md) — the lossless integer-matmul twin: no tile, the value-domain reference any physical mode is compared against.

The abstract contract these share — the tiling scheme, the value domain, and the ADC surface — is in [macro/base](../base.md).
