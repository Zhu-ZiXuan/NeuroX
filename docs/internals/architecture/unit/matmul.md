# QuantMatMul

The cross-unit software contract: the structural `QuantMatMul` Protocol that every unit implementation satisfies by shape. It fixes the surface a consumer calls — the accepted value ranges, the ADC operating-point publication, and the `program` / `matmul` / `fabricate` lifecycle — and owns none of the silicon its members aggregate.

## Design decisions

- **The contract is a structural `Protocol`, not a shared base class.** A unit satisfies `QuantMatMul` by shape, not by inheritance. Structural typing lets heterogeneous units — a degenerate exact-integer member and registry-organized crossbar members — present one consumer-facing surface without a forced common ancestor, so no member inherits a base it does not need.
- **The `QuantMatMul` contract carries no PPA surface.** The structural Protocol fixes the value-domain surface a consumer calls and stops there: area and leakage are absent by design, not by omission. Silicon is accounted on the module tree the profiler walks, an axis orthogonal to this contract — so what a unit reports is settled there, never by conforming here.
- **`[Sa, Sw, Tc, Tr]` is the fixed leading-axis order.** Every mode's organized weight tensor places its present slice/tile axes in this canonical order ahead of the tile-owned `(data, D, row)` trailing block. A mode that does not use an axis omits it entirely rather than padding it size-1 (the `M=1` / `Sa=1` placeholders inserted for broadcast against the activation are a separate matter). Fixing the order across modes is what lets the aggregate reductions name their axes by a stable negative index.

## Contracts & invariants

- **`program` is bound to one `w_logical_shape`.** The shape `(*prefix, N, K)` is fixed at construction and `program(weight)` rejects any other shape. A unit is single-weight: re-`program` overwrites, it does not re-shape.
- **`matmul` returns the pure integer dot product.** No bias and no rescale are folded in; bias add and the requantize back to the activation grid are the operator layer's responsibility, kept off the unit so its value-domain-only contract holds.
- **The ADC surface is published, not enforced.** `adc_mode_num` / `adc_max_bits` advertise the operating-point range; `adc_rescale_factor` raises `KeyError` for an uncalibrated point. `adc_max_bits == 0` is the "no output quantization" sentinel — not a one-level ADC. Value ranges (`w_value_range` / `x_value_range`) are likewise published capability, never a runtime bound — out-of-range inputs are not checked.
- **`fabricate` resamples static variation tree-wide.** The no-arg `fabricate()` re-samples static manufacturing variation across the unit and all its descendants in one call.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/matmul.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
