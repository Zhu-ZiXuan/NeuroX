# `IdealXbarMacro`

Degenerate XbarMacro family member, declared in `neurox/macro/xbar/ideal.py`. Lossless integer-matmul reference; serves as the ideal stand-in for any other XbarMacro under the same Protocol surface.

## Public surface

- `IdealXbarMacroConfig` — frozen dataclass with:
  - `x_value_range: tuple[int, int]`
  - `w_value_range: tuple[int, int]`
- `IdealXbarMacro` — registered via `@XbarMacro.register_key(IdealXbarMacroConfig)`.
  - Init signature matches the family root: `(*, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)`. `dtype`, `T__K`, and `ideal_xbar` are accepted for API uniformity and ignored.
  - `program(weight)` stores the integer weight tensor verbatim into `self.weight`.
  - `matmul(input)` does `torch.matmul(input.float(), weight.float().T)` and rounds back to the input dtype.
  - `output_rescale_factor == 1.0` (no analog gain to undo).

## When to use

- Smoke-testing the operator pipeline without dragging in analog parameters.
- Sanity baseline against which `XbarMacro + IdealXbarConfig` and `XbarMacro + physical xbar` results should converge in the no-noise limit.

## What it does **not** do

- No fabrication variation (no children to resample).
- No PPA contribution from internal modules — profiler aggregation is sparse.
- No bias add, no requantize — those live in the operator regardless of macro flavour.
