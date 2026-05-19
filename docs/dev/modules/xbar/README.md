# Xbar Modules

This directory documents the physical-crossbar layer.

The xbar layer owns physical-array semantics and exposes the primitive shape contract — `fabricate(w)` and `vec_mat_mul(x)` — plus a runtime `(adc_mode, adc_bits) → rescale_factor` lookup.

## Top-level files

- [`base.md`](base.md) — abstract `Xbar` + `XbarConfig` + rescale lookup contract.
- [`ideal.md`](ideal.md) — `IdealXbar`, the lossless tile-level reference.
- [`solver.md`](solver.md) — shared circuit-solver primitives (`elementwise_diff`, `solve_tridiagonal`).

## Concrete families

- [`_1t1r/`](_1t1r/README.md) — the 1T1R xbar stack (offset-coded today; differential / simple variants are reserved).

See also:

- `../../architecture/mapping.md`
