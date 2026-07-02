# IdealXbarMacro

## Summary

`IdealXbarMacro` (`xbar/ideal.py`): the degenerate family member that joins the `XbarMacro` registry without owning a tile. It stores the integer weight and runs `torch.matmul`. Spec: [reference/macro/xbar/ideal](../../../reference/macro/xbar/ideal.md).

## Design decisions

- **In the registry, but skips every layered step.** It registers via `@XbarMacro.register_key(IdealXbarMacroConfig)` so it shares the family factory and the `NeuroxMacroQuantMatMul` surface, but it declares no `xbar`, builds none (`_build_xbar` is never called), and carries an empty `IdealXbarMacroPolicy`. The family signature args `policy` / `dtype` / `T__K` / `ideal_xbar` are accepted for uniformity and ignored.
- **Sentinel ADC surface.** `adc_mode_num == 1`, `adc_max_bits == 0`, `adc_rescale_factor == 1.0`. The `0` bit count is the operator's "skip output quantization" signal — see [base](../base.md).
- **0-d nominal weight buffer.** `weight` and `nominal_weight` register as 0-d `int32` buffers (`persistent=False`), giving `weight` a defined attribute placeholder before any `program` call while keeping it out of the state dict.

## Contracts & invariants

- **`program` shape gate, then verbatim store.** `program(weight)` rejects any shape other than `w_logical_shape`, then stores the tensor unchanged into `self.weight` — no encoding, no slicing.
- **`matmul` widens to int64.** Both operands cast to `int64` before `torch.matmul(input, weight.T)` to avoid overflow at full integer width; the result is exact.

## Performance & resources

A single dense `int64` matmul; no tiling, no analog cost. No PPA contribution (no constituent circuits), so profiler aggregation over an ideal macro is sparse.

## Gotchas

- **Not a stand-in for a physical mode in a PPA study.** It has no children, so it contributes no area / leakage / energy; use it only as a value-domain reference, not an energy baseline.
- **Bring-up / reference only, never a production accuracy result.** Directly instantiating an ideal member (this degenerate macro, or a macro wrapping a directly built `IdealXbar`) yields a synthetic, uncalibrated reference; a hardware-faithful ideal twin comes from `to_ideal()` on a physical config.

## Known limitations

- N/A.

---

- **Reference**: [ideal](../../../reference/macro/xbar/ideal.md)
- **Implementation**: `neurox/macro/xbar/ideal.py`
- **Tests**: `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this module.
