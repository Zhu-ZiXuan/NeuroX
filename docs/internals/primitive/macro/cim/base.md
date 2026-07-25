# CimMacro base

`CimMacro` defines the primitive program/read, value-domain, and output-rescale surface shared by the family.

## Design decisions

- **Family dispatch keyed on config and policy types.** `(type(config), type(policy))` is the discriminator, so adding a member registers one concrete pair and never changes `from_config`; mismatched config-policy wiring fails before leaf construction.

## Contracts & invariants

- **Logical shape contract.** `from_config` passes owner-selected `input_num` and `output_num` to the concrete constructor. `program(w)` then takes `(*inst_shape, input_num, output_num)`; `vec_mat_mul(x)` takes primitive trailing `[input_num]` and returns primitive trailing `[output_num]`. Every leading axis of `x` is anonymous broadcast batch. The macro owns all logical-to-physical encoding and layout conversion.
- **Geometry belongs to the instance, not the config.** `CimMacroConfig` contains no dimensions. A physical member binds the constructor's logical dimensions to its own physical terminology immediately, such as `row_num = input_num` and `col_num = output_num`, and uses only that physical terminology afterward. The ideal member retains `input_num` and `output_num` because it has no physical row/column model. The abstract family deliberately exposes neither pair after construction.
- **Selection limit.** The base config requires a positive `max_active_num` and imposes no divisibility requirement. A concrete member validates the limit against its own logical-to-physical mapping. The caller masks unselected positions to zero. Selection is scheduling metadata and is independent of the runtime value at a selected position.
- **`_split_col_lanes(t, *, col_per_lane)` splits the trailing column axis into `(lane_num, col_per_lane)`.** The lane axis aligns with fabricated instance axes (parallel circuit copies) and the trailing axis is time-serial on each lane, with `lane = col // col_per_lane`; non-divisible extents raise `ValueError` — no clamping. Serial-vs-parallel semantics live in tensor shape: axes matching a stage's fabricated `inst_shape` are parallel circuit copies, every other axis is time-serial on that hardware.
- **Abstract value-domain and ADC surface.** Each subclass implements the logical value-domain properties (`x_value_range`, `w_value_range`) and the ADC operating-point surface (`adc_mode_num`, `adc_max_bits`, `adc_rescale_factor`). Internal digit geometry is deliberately absent from the base API. `adc_rescale_factor(*, adc_mode, adc_bits)` returns the code-to-dot-product rescale keyed on the `(adc_mode, adc_bits)` pair and raises `KeyError` for an uncalibrated operating point.
- **`to_ideal()` preserves the instance geometry and family contract.** The base stores the constructor's logical dimensions privately as one shape tuple and uses the abstract value-domain and ADC properties to construct the ideal counterpart. Physical members do not repeat this logic.
- **Ownership.** Fabricated state lives in the device children; the tile owns no static mismatch, so `_sample_fabricate_mismatch` is an explicit base no-op and the cascade fans into the children.

---

- **Reference**: [CIM macro family](../../../../reference/primitive/macro/cim/family.md)
- **Implementation**: `neurox/primitive/macro/cim/base.py`
- **Tests**: TODO — the base contract is exercised through family members
