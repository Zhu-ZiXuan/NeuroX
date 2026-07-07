# Macro base

## Summary

The cross-macro contract: the structural `NeuroxMacroQuantMatMul` Protocol (`base.py`) that every macro impl satisfies, and the project-wide leading-axis layout convention the modes share. The concrete family lives under xbar/.

## Design decisions

- **Contract is a structural `Protocol`, not a base class.** `NeuroxMacroQuantMatMul` is duck-typed: a macro satisfies it by shape, not by inheritance. This lets the degenerate ideal member and the xbar-family members share one consumer-facing surface without forcing the ideal macro under the `XbarMacro` registry base it does not need. The registry root `XbarMacro` is a separate concrete base under xbar/; the two are deliberately not the same type.
- **A macro is not a `CircuitBase`.** Macros own no silicon, so they carry no `CircuitConfig`-backed PPA; `area_per_inst__um2` and `leakage_per_inst__uW` return zero and PPA appears in reports only through the constituent circuits (tiles, reducers). Making a macro a circuit would double-count or invent area it does not have.
- **`[Sa, Sw, Tc, Tr]` is the fixed leading-axis order.** Every mode's organized weight tensor places its present slice/tile axes in this canonical order ahead of the tile-owned `(data, D, row)` trailing block. A mode that does not use an axis omits it entirely rather than padding it size-1 in the layout convention (the `M=1`/`Sa=1` placeholders inserted for broadcast against the activation are a separate matter). Fixing the order across modes is what lets the aggregate reductions name their axes by a stable negative index.

## Contracts & invariants

- **`program` is bound to one `w_logical_shape`.** The shape `(*prefix, N, K)` is fixed at construction and `program(weight)` rejects any other shape. A macro is single-weight: re-`program` overwrites, it does not re-shape.
- **`matmul` returns pre-requantize int.** The contract matches `torch.matmul` (pure dot product, no bias, no rescale). Bias add and requantization lie outside the macro; folding them in would break the value-domain-only contract.
- **The ADC surface is published, not enforced.** `adc_mode_num` / `adc_max_bits` advertise the operating-point range; `adc_rescale_factor` raises `KeyError` for an uncalibrated point. `adc_max_bits == 0` is the "no output quantization" sentinel — not a one-level ADC. Value ranges (`w_value_range` / `x_value_range`) are likewise published capability, never a runtime bound — out-of-range inputs are not checked.
- **`fabricate` resamples static variation tree-wide.** The no-arg `fabricate()` re-samples static manufacturing variation across the macro and all its descendants in one call.

---

- **Reference**: [macro base](../../reference/macro/family.md)
- **Implementation**: `neurox/macro/base.py`
- **Tests**: `tests/test_xbar_macro.py`
