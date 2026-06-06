# `neurox/xbar/ideal.py`

## Current role

`IdealXbar` is the lossless tile-level reference for any physical xbar. It preserves the same primitive shape contract and the same output-rescale grid as the physical xbar, but discards every analog non-ideality (IR drop, noise, finite gain, …). Any code written against the generic `Xbar` surface accepts it unchanged.

## What "ideal" means

The "ideal" in `IdealXbar` refers to the **analog side**:

- analog physics: absent
- integer-output grid (`adc_rescale_factor`): preserved
- input contract (digit tensor shape, value ranges): preserved

The same per-tile xbar-native digit tensor feeds a physical xbar or its `to_ideal()` twin without modification.

## Lifecycle

1. `__init__(*, cfg, name, inst_shape, dtype, T__K)` registers a 0-d nominal `nominal_digits=0` buffer, mirrors it onto `digits`, and pre-computes the LSB-first `digit_weights` LUT. The per-instance multiplicity is committed via `inst_shape`; the full digit-tensor shape `self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)` is derived inside the xbar.
2. `_sample_fabricate_mismatch` is the inherited no-op — `IdealXbar` has no static mismatch.
3. `program(w)` validates `w.shape == self._w_layout_shape` and reassigns the `digits` buffer to `w`.
4. `vec_mat_mul(x, *, adc_operation_point)` widens digits / weights / activations to `int64`, collapses the digit axis with the radix-derived `digit_weights` vector `(1, r, r², …, r^(D-1))`, runs an exact integer dot-product, and — for `adc_operation_point.adc_bits > 0` — floor-quantises as `code = floor(M_ideal / rescale_factor)` then clamps to the signed `adc_bits` range. The quantize-side reciprocal `1 / rescale_factor` is pre-computed at `__init__` into `_scale_lut` and applied as a multiplication at runtime. `adc_bits == 0` is a sentinel: the lossless integer dot product is returned unmodified, matching `IdealXbarMacro`.

Until `program(...)` is first called, `digits` broadcasts the 0-d nominal `0` against any activation, producing the nominal "no-weight" output. The ideal tile knows nothing about offset coding, reference columns, or any analog-domain digit-shift-add — every encoding detail lives in the physical xbar that produced it.

## How to obtain one

Two equivalent paths, both producing the same lossless tile:

- **From a physical tile**: `physical.to_ideal()`. The base `Xbar.to_ideal` builds an `IdealXbarConfig` from the source xbar's `XbarConfig` fields plus its abstract `x_range` / `w_digit_*` properties and its `adc_mode_num` / `adc_max_bits` surface, then instantiates `IdealXbar` with the source tile's `inst_shape`. Macros use this path when their build-time `ideal_xbar=True` toggle is set.
- **Standalone**: `Xbar.from_config(cfg=IdealXbarConfig(...), ...)` (or, equivalently, the direct `IdealXbar(cfg=IdealXbarConfig(...), ...)` constructor). `IdealXbarConfig` carries the structural fields (`x_range`, `w_digit_count`, `w_digit_radix`, `w_digit_range`) plus the ADC operating-point metadata (`adc_mode_num`, `adc_max_bits`) that an ideal tile cannot derive from a base `XbarConfig` alone. This path is the only way to materialise an ideal tile when no physical twin exists.

`IdealXbar.to_ideal()` returns `self` — an ideal tile is its own ideal counterpart.

## When to use the ideal twin

- **Tests**: every xbar test that compares the physical result to an integer truth uses the ideal twin as reference.
- **Calibration**: the ADC-boundary tool runs the physical and ideal xbars against the same random inputs and uses the gap to pick comparator thresholds.

`IdealXbar` is a **tile-level** reference: it bypasses analog physics but still publishes the full xbar capability surface. A lossless reference at coarser granularity belongs at that coarser layer, not here.

See also:

- `base.md`
- `docs/dev/modules/xbar/_1t1r/offset.md`
