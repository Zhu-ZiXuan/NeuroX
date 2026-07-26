# Conv2dCimUnit

The engine-backed CIM unit exposing the conv2d operator:
`Conv2dUnit` × `EngineBackedCimUnit`. It lowers the operator to one logical
matmul and delegates physical placement to the configured engine.

## Design decisions

- **One programmed kernel matrix.** `_weight_to_matrix` flattens
  `[C_out, C_in, kh, kw]` directly to `[C_out, K]`. It does not construct a
  Toeplitz matrix or replicate weights for different spatial windows.
- **Windows are runtime work.** `_conv2d_planes` gathers every convolution
  window as one row of `[..., M, K]`. The `M=H_out*W_out` axis rides through
  `engine.matmul` as runtime serial work against the same programmed matrix.
- **The engine owns physical mapping.** Input-axis block packing, contraction
  tiling, activation phases, precision slicing, macro execution, and output
  block restoration are engine concerns. The conv2d unit neither reads macro
  geometry nor constructs physical row/column layouts.
- **Geometry remains operator-owned.** `stride`, `padding`, and `dilation` are
  configuration fields. Kernel extent comes from the 4-D
  `w_logical_shape`.
- **Padding is the only injected activation zero.** Non-zero padding requires
  `0` in the engine's activation range. Window gathering introduces no surplus
  windows or strip-fill values.

## Contracts & invariants

- `w_logical_shape` is exactly `(C_out, C_in, kh, kw)` with no weight-batch
  prefix.
- The engine logical weight shape is
  `(C_out, C_in*kh*kw)`.
- Weight and window flattening use the same
  `(C_in, kh, kw)` row-major order.
- `program(weight, bias=None)` programs one flattened weight matrix and the
  optional `(C_out,)` integer bias.
- `_conv2d_fold` restores `M` directly to `(H_out,W_out)` and moves
  `C_out` to the trailing operator channel position.

---

- **Reference**: [conv2d mapping](../../../../reference/architecture/unit/conv2d.md), [engine family](../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/conv2d.py`
- **Tests**: `tests/architecture/unit/test_conv2d_cim_unit.py`
