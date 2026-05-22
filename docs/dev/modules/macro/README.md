# Macro Modules

`neurox/macro/` is the orchestration layer that wraps one xbar tile with mode-specific organize / aggregate logic to produce a quantised-integer matmul.

## Public surface

- `NeuroxMacroQuantMatMul` — structural Protocol every macro impl satisfies. Method surface: `fabricate()` (no-arg static-mismatch resample), `program(weight)` (write the static weight state), `matmul(input, *, adc_operation_point)` (forward against the programmed state at the runtime ADC operating point — pure int matmul, matches `torch.matmul`; bias and requantize live in the operator), `adc_rescale_factor(adc_operation_point)` (rescale-factor lookup). Property surface: `w_value_range`, `x_value_range`, `adc_mode_num`, `adc_max_bits`.
- [`xbar/`](xbar/README.md) — the XbarMacro family lives entirely under this subpackage. `XbarMacro` is the abstract registry root; `XbarMacro.from_config(cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)` is the single factory entry. Concrete members: `DirectXbarMacro`, `InterArraySliceXbarMacro`, `IntraArraySliceXbarMacro` (xbar-using; each declares its own `xbar_cfg`), plus the degenerate `IdealXbarMacro` (no xbar; lossless integer-matmul reference, [doc](xbar/ideal.md)).

## Architecture rules

See [`docs/dev/architecture/xbar_macro.md`](../../architecture/xbar_macro.md) for:

- W-side `slice → organize → tile` decomposition.
- X-side `slice → tile` decomposition.
- Organize ↔ aggregate duality (every mode owns paired halves).
- `Sw` layout vs `Sa` schedule semantics.
- How to add a new mode subclass.

## Integer-grid capability

Each macro exposes `w_value_range` / `x_value_range` properties describing the integer ranges it accepts. Subclasses delegate to the slicer they own.
