# Mapper (Value-Domain Mapping)

The value-domain mapping math: how logical weights and activations are carried into the physical integer value domain of a tile, independent of any analog physics. Two layers compose here — the encoding of a single integer into a fixed-length signed-digit string, and the decomposition of a tensor into per-slice digit slots.

- [transcoder/](transcoder/README.md) — the signed-digit encoding policies (true-form, radix-complement, canonical) and the positional decode.
- [xbar/](xbar/README.md) — the slicer family that decomposes an integer tensor into `[slice_num, digit_num]` for an xbar's primitive cell grid.
