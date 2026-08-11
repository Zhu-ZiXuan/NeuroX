# Ideal CIM macro

## Design decisions

- **Direct registry member.** `IdealCimMacro` registers its config-policy pair and can
  be built either directly or as the ideal counterpart of another family
  member.
- **Window-driven per-mode conversion.** Each quantization mode owns one
  canonical inclusive MAC-unit window `[lower, upper]` in
  `quantization_input_ranges`, indexed by `quantization_mode`. `vec_mat_mul`
  quantizes every WL plane independently against the selected window,
  preserving quantize-then-accumulate semantics. The conversion reproduces the
  unsigned reading a converter performs and then subtracts the window's zero
  code, so the returned codes are signed.
- **Lossless sentinel.** `adc_bits is None` bypasses quantization and returns
  the integer plane dot unmodified. It still validates `quantization_mode`
  against the declared windows; `adc_bits` is not checked against
  `adc_max_bits` in this branch, since the sentinel sits outside that bound by
  construction (`adc_max_bits >= 1` is required at config time).
- **All-integer conversion.** `_convert` never forms the fractional step
  `W / 2^adc_bits`. It floor-divides `(plane_dot - lower) * 2^adc_bits`
  by the window width `W = upper - lower + 1` in int64 with
  `rounding_mode="floor"`, so a negative dot floors toward $-\infty$ rather
  than truncating toward zero. Training mode adds a uniform integer jitter on
  `[0, W - 1]` to the numerator before the floor, an unbiased stochastic
  rounding of the floor remainder; because the window is canonical the jitter
  is zero-preserving: an on-grid dot — the window zero included — takes the
  same code in training and evaluation mode.
- **Config validates the window shape alone.** Canonical
  shapes fix the zero code at `0` for an unsigned window and `2^(adc_bits - 1)`
  for a mid-zero one, an integer at every bit width by construction, so the
  runtime computes it instead of reading it and never rounds the zero-code
  subtraction. A window is not required to be a power of two, and one with
  `W > 2^adc_bits` is a legal lossy operating point.
- **Bound-selected arithmetic path.** A plane-dot bound below $2^{24}$ selects
  fp32 `einsum`; the bound guarantees exact integer representation for
  range-conformant operands with at most `max_active_num` selected inputs. Larger
  bounds use an int64 multiply-reduce path.
- **The rescale anchor.** `_max_bits_rescale_factor` returns `1.0` — these
  codes are the currency every other member's factor is expressed in — and
  `to_ideal` returns `self`.

## Contracts & invariants

- `program` clones one logical integer matrix with shape
  `(*inst_shape, input_num, output_num)` into `_w`. The input already carries
  the intended device; module migration must precede programming.
- `vec_mat_mul` preserves leading-axis order and returns trailing
  `[output_num]`. The result is `int64` for both the lossless and quantized
  branches — an all-positive window at high `adc_bits` produces unsigned
  codes up to `2^adc_bits - 1`, which overflows a 16-bit type, so both
  branches share one dtype.
- Training mode uses stochastic floor quantization; evaluation mode uses
  deterministic floor quantization. Both clamp the unsigned code to
  `[0, 2^adc_bits - 1]` before the zero-code offset.
- `quantization_mode` is validated unconditionally, including for the lossless
  oracle. A mode outside `[0, len(quantization_input_ranges))`, or `adc_bits`
  neither `None` nor in `[1, adc_max_bits]`, raises `ValueError`.
- **Bit widths nest.** In evaluation mode the unsigned code at `adc_bits = b`
  is the unsigned code at `adc_max_bits` right-shifted by `adc_max_bits - b` —
  the arithmetic twin of a physical converter truncating its decision cycles
  over one shared reference ladder, so the ideal macro tracks a physical member
  across bit widths rather than defining its own ladder per width.
- **A twin of a sign-magnitude member is unfaithful by design.** A mid-zero
  window carries one phantom bottom level and a single zero, where a
  sign-magnitude encoding has no bottom level and a double zero; that gap is
  part of the physical-vs-ideal difference, so a twin comparison is never
  bit-exact at a finite bit width.
- The policy is empty, and static PPA and duration are alike zero: an
  arithmetic oracle fabricates no silicon and has no circuit to take time, so
  `latency__ns` returns `0.0` at every resolution.

## Performance & resources

The fp32 path keeps the row contraction GPU-capable. The int64 path preserves
exactness when the fp32 bound is not satisfied.

---

- **Reference**: [ideal CIM macro](../../../../reference/primitive/macro/cim/ideal.md)
- **Implementation**: `neurox/primitive/macro/cim/ideal.py`
- **Tests**: `tests/primitive/macro/test_ideal_cim_macro_rescale.py`, `tests/primitive/macro/test_ideal_cim_macro_phase_semantics.py`, `tests/primitive/macro/test_ideal_cim_macro_fp32_exact.py`
