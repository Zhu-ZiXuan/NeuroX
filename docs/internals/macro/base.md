# Macro base

## Summary

The cross-macro contract: the structural `NeuroxMacroQuantMatMul` Protocol (`base.py`) that every macro impl satisfies, and the project-wide leading-axis layout convention the modes share. The concrete family lives under [xbar/](xbar/README.md). Spec: [reference/macro/base](../../reference/macro/base.md).

## Design decisions

- **Contract is a structural `Protocol`, not a base class.** `NeuroxMacroQuantMatMul` is duck-typed: a macro satisfies it by shape, not by inheritance. This lets the degenerate ideal member and the xbar-family members share one consumer-facing surface without forcing the ideal macro under the `XbarMacro` registry base it does not need. The registry root `XbarMacro` is a separate concrete base under [xbar/](xbar/README.md); the two are deliberately not the same type.
- **A macro is not a `CircuitBase`.** Macros own no silicon, so they carry no `CircuitConfig`-backed PPA; `area_per_inst__um2` and `leakage_per_inst__uW` return zero and PPA appears in reports only through the constituent circuits (tiles, reducers). Making a macro a circuit would double-count or invent area it does not have.
- **`[Sa, Sw, Tc, Tr]` is the fixed leading-axis order.** Every mode's organized weight tensor places its present slice/tile axes in this canonical order ahead of the tile-owned `(data, D, row)` trailing block. A mode that does not use an axis omits it entirely rather than padding it size-1 in the layout convention (the `M=1`/`Sa=1` placeholders inserted for broadcast against the activation are a separate matter — see [xbar/base](xbar/base.md)). Fixing the order across modes is what lets the aggregate reductions name their axes by a stable negative index.

## Contracts & invariants

- **`program` is bound to one `w_logical_shape`.** The shape `(*prefix, N, K)` is fixed at construction and `program(weight)` rejects any other shape. A macro is single-weight: re-`program` overwrites, it does not re-shape.
- **`matmul` returns pre-requantize int.** The contract matches `torch.matmul` (pure dot product, no bias, no rescale). Bias add and requantization are the operator's job; a macro that folded them in would break the value-domain-only contract.
- **The ADC surface is published, not enforced.** `adc_mode_num` / `adc_max_bits` advertise the operating-point range; `adc_rescale_factor` raises `KeyError` for an uncalibrated point. Value ranges (`w_value_range` / `x_value_range`) are likewise published capability, never a runtime bound — out-of-range inputs are not checked.
- **`fabricate` resamples static variation tree-wide.** The no-arg `fabricate()` re-samples static manufacturing variation across the macro and all its descendants in one call; it takes no arguments and returns nothing.

## Performance & resources

N/A at this level — the base stores only the construction context (`_macro_dtype`, `_macro_T__K`, `_ideal_xbar`, `_macro_name`, `_w_logical_shape`). The memory- and compile-sensitive work is in the concrete modes under [xbar/](xbar/README.md).

## Gotchas

- **`adc_max_bits == 0` is a sentinel, not a degenerate ADC.** Zero means "no output quantization"; it propagates into the operator's default `AdcOperationPoint`. Do not treat it as a one-level ADC.

## Known limitations

- N/A.

---

- **Reference**: [macro base](../../reference/macro/base.md)
- **Implementation**: `neurox/macro/base.py`
- **Tests**: `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this module.
