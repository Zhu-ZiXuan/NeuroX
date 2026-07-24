# Ideal CIM macro

## Design decisions

- **Direct registry member.** `IdealCimMacro` registers its config type and can
  be built either directly or as the ideal counterpart of another family
  member.
- **Per-plane quantization.** `vec_mat_mul` uses `adc_bits` and treats
  `adc_mode` as opaque. Each WL plane is quantized independently, preserving
  quantize-then-accumulate semantics.
- **Lossless sentinel.** `adc_bits == 0` bypasses quantization and returns the
  integer plane dot. Width one is absent from the rescale table because the
  signed symmetric denominator is zero.
- **Bound-selected arithmetic path.** A plane-dot bound below $2^{24}$ selects
  fp32 `einsum`; the bound guarantees exact integer representation for
  range-conformant inputs with at most `active_row_num` live rows. Larger
  bounds use an int64 multiply-reduce path.

## Contracts & invariants

- `program` stores one integer digit tensor matching `_w_layout_shape` as ordinary programmed state. The input already carries the intended device; module migration must precede programming.
- `vec_mat_mul` preserves leading-axis order and returns trailing
  `[col_num]`.
- Training mode uses stochastic floor quantization; evaluation mode uses
  deterministic floor quantization. Both clamp to the signed code range.
- The policy is empty and static PPA is zero.

## Performance & resources

The fp32 path keeps the row contraction GPU-capable. The int64 path preserves
exactness when the fp32 bound is not satisfied.

---

- **Reference**: [ideal CIM macro](../../../../reference/primitive/macro/cim/ideal.md)
- **Implementation**: `neurox/primitive/macro/cim/ideal.py`
- **Tests**: `tests/primitive/macro/test_ideal_cim_macro_rescale.py`, `tests/primitive/macro/test_ideal_cim_macro_fp32_exact.py`
