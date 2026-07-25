# CimMacro base

`CimMacro` defines the primitive program/read, value-domain, and output-rescale surface shared by the family.

## Design decisions

- **Family dispatch keyed on config and policy types.** `(type(config), type(policy))` is the discriminator, so adding a member registers one concrete pair and never changes `from_config`; mismatched config-policy wiring fails before leaf construction.

## Contracts & invariants

- **Primitive shape contract.** `program(w)` takes `(*inst_shape, col_num, w_digit_count, row_num)`; `vec_mat_mul(x)` takes WL planes with primitive trailing `[row_num]` and returns codes with the same leading order and primitive trailing `[col_num]`. Every leading axis of `x` is anonymous broadcast batch. Rows outside the active window must be zero. The macro neither creates nor reduces a phase axis.
- **`active_row_num` is range-guarded only.** `CimMacroConfig.validate()` enforces `1 <= active_row_num <= row_num` and imposes no divisibility requirement. `max_active_num` exposes the configured limit.
- **`_split_col_lanes(t, *, col_per_lane)` splits the trailing column axis into `(lane_num, col_per_lane)`.** The lane axis aligns with fabricated instance axes (parallel circuit copies) and the trailing axis is time-serial on each lane, with `lane = col // col_per_lane`; non-divisible extents raise `ValueError` — no clamping. Serial-vs-parallel semantics live in tensor shape: axes matching a stage's fabricated `inst_shape` are parallel circuit copies, every other axis is time-serial on that hardware.
- **Abstract value-domain and ADC surface.** Each subclass implements the value-domain properties (`x_value_range`, `w_digit_count`, `w_digit_radix`, `w_digit_value_range`) and the ADC operating-point surface (`adc_mode_num`, `adc_max_bits`, `adc_rescale_factor`). `adc_rescale_factor(*, adc_mode, adc_bits)` returns the code-to-dot-product rescale keyed on the `(adc_mode, adc_bits)` pair passed as two keyword-only ints, and raises `KeyError` for an uncalibrated operating point.
- **`to_ideal()` preserves the family contract.** The counterpart carries the base geometry and value-domain fields; implementation-specific calibration state is excluded.
- **Ownership.** Fabricated state lives in the device children; the tile owns no static mismatch, so `_sample_fabricate_mismatch` is an explicit base no-op and the cascade fans into the children.

---

- **Reference**: [CIM macro family](../../../../reference/primitive/macro/cim/family.md)
- **Implementation**: `neurox/primitive/macro/cim/base.py`
- **Tests**: TODO — the base contract is exercised through family members
