# `neurox/mapper/xbar/simple_mapper.py`

## Current role

`SimpleMapper` is the current default `XbarMapper` implementation. It composes:

- `SimpleTiler` — matrix decomposition with automatic padding.
- a configured `Slicer` for activations (`SerialSlicer` by default — serial bit-cycle decomposition).
- a configured `Slicer` for weights (`SimpleSlicer` by default — direct digitise-then-group).

Both slicers carry their own encoding string (true-form / complement / canonical) and digit-count / digit-radix triple; the mapper does not see those policy bits directly.

## Construction

`SimpleMapper(tiler, x_slicer, w_slicer)` — receives pre-built tiler and slicer instances. The mapper does **not** own slicer encoding configuration; that policy lives outside the mapper and is injected through the constructed slicers.

## Forward path

`fabricate(weights)`:

1. tile the weight matrix to fit the xbar's `(col_num, row_num)` tile
2. slice each tile with the w slicer into `[..., data_num, digit_num, row_num]`
3. hand the digit tensor to `Xbar.fabricate(...)`

`forward(activations)`:

1. tile the activation to match the row dimension;
2. slice with the x slicer into `[..., slice_num, digit_num, row_num]`;
3. call `Xbar.vec_mat_mul(x)` per slice cycle.

The mapper yields one xbar output per slice cycle; cross-cycle aggregation is performed above the mapper layer.

See also:

- `base.md`
- `docs/dev/modules/mapper/xbar/slicer/README.md`
- `docs/dev/modules/mapper/xbar/tiler/README.md`
- `docs/dev/architecture/mapping.md`
