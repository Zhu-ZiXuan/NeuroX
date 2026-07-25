# Ideal CIM macro

## Design decisions

- **Direct registry member.** `IdealCimMacro` registers its config-policy pair and can
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
  range-conformant operands with at most `max_active_num` selected inputs. Larger
  bounds use an int64 multiply-reduce path.

## Contracts & invariants

- `program` clones one logical integer matrix with shape
  `(*inst_shape, input_num, output_num)` into `_w`. The input already carries
  the intended device; module migration must precede programming.
- `vec_mat_mul` preserves leading-axis order and returns trailing
  `[output_num]`.
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
