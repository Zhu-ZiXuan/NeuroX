# Tiler Family

Tilers handle **geometry decomposition** only — they take a matrix and decompose it into the xbar's per-tile `(col_num, row_num)` blocks, with automatic padding when the input dimensions are not multiples of the tile shape.

Value decomposition is the [`slicer`](docs/dev/modules/mapper/xbar/slicer/README.md)'s job; tilers do not look at integer values.

## Generic interface

Concrete tilers accept the xbar's tile geometry and return the per-tile view of the input matrix plus the per-tile prefix axes (`Tc`, `Tr`) that identify which tile is which.

## Concrete implementations

- `SimpleTiler` — straightforward block decomposition with right / bottom zero-padding. The current default for offset-coded 1T1R deployments.

Future tilers (`StridedTiler`, `BlockSparseTiler`, …) will plug into the same surface.

See also:

- `docs/dev/modules/mapper/xbar/base.md`
- `docs/dev/architecture/mapping.md`
