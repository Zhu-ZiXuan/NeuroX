# Macro Modules

`neurox/macro/` is the orchestration layer that wraps one xbar tile (or its ideal twin) with the surrounding mapper + digital pipeline. Each macro instance corresponds to one NeuroX "macro" in the chip — one tile plus the digital aggregation that turns per-cycle xbar codes into one operator output tensor.

## Public surface

- `NeuroxMacroQuantMatMul` — the structural `Protocol` every macro implementation satisfies. A `Protocol`-typed handle allows the concrete macro to be swapped without changes outside the macro layer.
- `XbarMacro` — the production macro. Owns one xbar (physical or ideal), one `XbarMapper`, and the four digital blocks (column accumulator, weight shift-adder, activation shift-adder, requantizer).
- `IdealMacro` — a lossless reference that skips tiling. Useful in tests and ranges where the integer truth is needed without going through the mapping pipeline.

## Mapping flow

The macro owns one mapper (see [`architecture/mapping.md`](../../architecture/mapping.md)). The mapper holds the tiler and both slicers; the macro itself does not split weights or activations directly.

Per-call:

1. Mapper tiles + slices the activations and weights into the xbar's primitive shape.
2. The xbar runs `vec_mat_mul` per slice cycle.
3. Column accumulator + shift-adders combine per-cycle integer outputs into one logical-precision result.
4. Requantizer folds the xbar's analog rescale factor into the integer output grid the macro exposes.

## Integer-grid capability

Macros expose `w_value_range` / `x_value_range` properties describing the integer ranges they can carry. The values reflect the encoding actually supported by the xbar + mapper combo and are published as part of the macro's capability surface.

See also:

- `../xbar/README.md`
- `../mapper/README.md`
- `../digital/README.md`
- `../../architecture/mapping.md`
