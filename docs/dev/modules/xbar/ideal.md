# `neurox/xbar/ideal.py`

## Current role

`IdealXbar` is the lossless tile-level reference for any physical xbar. It preserves the same primitive shape contract and the same output-rescale grid as the physical xbar, but discards every analog non-ideality (IR drop, noise, finite gain, …). Any code written against the generic `Xbar` surface accepts it unchanged.

## What "ideal" means

The "ideal" in `IdealXbar` refers to the **analog side**:

- analog physics: absent
- integer-output grid (`output_rescale_factor`): preserved
- input contract (digit tensor shape, value ranges): preserved

The same per-tile xbar-native digit tensor feeds a physical xbar or its `to_ideal()` twin without modification.

## Convert pipeline

1. `fabricate(w)` stores the xbar-native digit tensor verbatim.
2. `vec_mat_mul(x)` collapses the digit axis with the radix-derived `digit_weights` vector `(1, r, r², …, r^(D-1))`, runs an exact integer dot-product, and floor-quantises to the configured `output_rescale_factor` grid.

The ideal tile knows nothing about offset coding, reference columns, or any analog-domain digit-shift-add — every encoding detail lives in the physical xbar that produced it.

## When to use the ideal twin

- **Tests**: every xbar test that compares the physical result to an integer truth uses the ideal twin as reference. `xbar.to_ideal()` is the canonical way to obtain it.
- **Calibration**: the ADC-boundary tool runs the physical and ideal xbars against the same random inputs and uses the gap to pick comparator thresholds.

`IdealXbar` is a **tile-level** reference: it bypasses analog physics but still publishes the full xbar capability surface. A lossless reference at coarser granularity belongs at that coarser layer, not here.

See also:

- `base.md`
- `docs/dev/modules/xbar/_1t1r/offset.md`
