# Macro Modules

`neurox/macro/` is the orchestration layer that wraps one xbar tile with mode-specific organize / aggregate logic to produce a quantised-integer matmul.

## Public surface

- `NeuroxMacroQuantMatMul` — structural Protocol every macro impl satisfies. Method surface: `fabricate()` (no-arg static-mismatch resample), `program(weight)` (write the static weight state), `matmul(input, bias, mult, rshift, zp)` (forward against the programmed state — no weight argument). Property surface: `w_value_range / x_value_range / output_rescale_factor`.
- `IdealMacro` — lossless reference matmul (no xbar, no mapping). Sibling to xbar macros under the same Protocol. Owns a buffer-backed `weight` written by `program(...)`.
- [`xbar/`](xbar/README.md) — xbar-backed family: `XbarMacro` + concrete modes (`InterXbarSliceMacro`, `IntraXbarSliceMacro`).

## Architecture rules

See [`docs/dev/architecture/xbar_macro.md`](../../architecture/xbar_macro.md) for:

- W-side `slice → organize → tile` decomposition.
- X-side `slice → tile` decomposition.
- Organize ↔ aggregate duality (every mode owns paired halves).
- `Sw` layout vs `Sa` schedule semantics.
- How to add a new mode subclass.

## Integer-grid capability

Each macro exposes `w_value_range` / `x_value_range` properties describing the integer ranges it accepts. Subclasses delegate to the slicer they own.
