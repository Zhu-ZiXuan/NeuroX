# xbar base — Implementation

## Summary

The crossbar family: the abstract `Xbar` (`base.py`) and the lossless `IdealXbar` (`ideal.py`). Concrete operating xbars live under [_1t1r/](_1t1r/README.md). Spec: [reference/xbar/base](../../reference/xbar/base.md).

## Design decisions

- **Family dispatch keyed on config type, not an `isinstance` ladder.** `type(config)` is the discriminator, so adding a concrete xbar only adds a `@Xbar.register_key(...)` line and never touches `from_config`. The cost — a config class per impl — is paid once.

## Contracts & invariants

- **Primitive shape contract.** `program(w)` takes `(*inst_shape, col_num, w_digit_count, row_num)`; `vec_mat_mul(x)` takes trailing `[row_num]` and returns trailing `[data_num]`. A caller's leading dims express **external batch and inst alignment only** — per-column / per-digit / per-phys_col fanout is an internal axis the implementation opens itself (`unsqueeze(-2)` against the fabricated `g` grid). Encoding a column/digit position in `x`'s leading dims is a contract violation.
- **`to_ideal()` carries `XbarConfig` fields only** — never `adc_calibration` (that lives on the physical config); the ideal twin must derive its rescale from geometry alone.
- **`IdealXbar` is also directly config-dispatchable** (it registers its own config key) — a convenience for flow bring-up and standalone tests. A directly built twin is hand-parameterised, bound to no fabricated device, and uncalibrated, so it is a synthetic reference only; production accuracy / PPA must use a twin from `to_ideal()` on a physical config.
- **Ownership.** Fabricated state lives in the device children; the xbar owns no static mismatch, so `_sample_fabricate_mismatch` stays the inherited no-op and the cascade fans into the children.

## Performance & resources

N/A at this level — the memory- and compile-sensitive work is in [_1t1r/](_1t1r/README.md).

## Gotchas

- **Silent broadcast flip.** If the size-1 inst slot is omitted from `x` and `x_batch` happens to equal an inst dim `K`, PyTorch broadcasts that position as a *matched* axis ("one x per inst") instead of the intended Cartesian "x_batch × inst". No error is raised; the output shape and the physical workload differ. Always insert the explicit `1` slot for Cartesian broadcast, even when `x_batch == K` coincides.

## Known limitations

- N/A.

---

- **Reference**: [xbar base](../../reference/xbar/base.md)
- **Implementation**: `neurox/xbar/base.py`, `neurox/xbar/ideal.py`
- **Tests**: `tests/test_xbar_physics.py`
- **Decisions**: N/A — no ADR governs this module.
