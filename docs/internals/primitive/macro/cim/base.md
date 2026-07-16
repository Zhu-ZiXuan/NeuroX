# CimMacro base

The crossbar family: the abstract `CimMacro` (`base.py`) and the lossless `IdealCimMacro` (`ideal.py`). The pure array is under [array/](../../xbar/README.md); concrete scheme xbars live in their own per-scheme packages.

## Design decisions

- **Family dispatch keyed on config type, not an `isinstance` ladder.** `type(config)` is the discriminator, so adding a concrete xbar only adds a `@CimMacro.register_key(...)` line and never touches `from_config`. The cost — a config class per impl — is paid once.

## Contracts & invariants

- **Primitive shape contract.** `program(w)` takes `(*inst_shape, col_num, w_digit_count, row_num)`; `vec_mat_mul(x)` takes trailing `[row_num]` and returns trailing `[col_num]`. A caller's leading dims express **external batch and inst alignment only** — per-column / per-digit / per-phys_col fanout is an internal axis the implementation opens itself (`unsqueeze(-2)`). Encoding a column/digit position in `x`'s leading dims is a contract violation. The caller must also insert an explicit size-1 inst slot in `x` for the Cartesian `x_batch × inst` broadcast, even when `x_batch` coincidentally equals an inst dim `K`: omit that slot and PyTorch silently broadcasts the position as a *matched* axis ("one x per inst") instead of the intended Cartesian product, raising no error while the output shape and the physical workload diverge.
- **Abstract value-domain and ADC surface.** Each subclass implements the value-domain properties (`x_range`, `w_digit_count`, `w_digit_radix`, `w_digit_range`) and the ADC operating-point surface (`adc_mode_num`, `adc_max_bits`, `adc_rescale_factor`). `adc_rescale_factor(adc_operation_point)` returns the code-to-dot-product rescale keyed on the `(adc_mode, adc_bits)` pair, and raises `KeyError` for an uncalibrated operating point.
- **`to_ideal()` carries `CimMacroConfig` fields only** — never a physical subclass's calibration table, which lives on the physical config; the ideal twin must derive its rescale from geometry alone.
- **`IdealCimMacro` is also directly config-dispatchable** (it registers its own config key) — a convenience for flow bring-up and standalone tests. A directly built twin is hand-parameterised, bound to no fabricated device, and uncalibrated, so it is a synthetic reference only; production accuracy / PPA must use a twin from `to_ideal()` on a physical config.
- **`IdealCimMacro` output-quantization contract.** `vec_mat_mul` consumes only `adc_operation_point.adc_bits`; `adc_mode` is opaque and never read. `adc_bits == 0` is the lossless sentinel — the integer dot product is returned unmodified, with ADC quantization and the signed clamp both bypassed. `adc_bits == 1` is unsupported — absent from the rescale table, so `adc_rescale_factor` raises a natural `KeyError` at that width. At `adc_bits >= 2` the quantizer follows `self.training`: under `train()` `stochastic_floor_to_int` adds a per-element `u ~ U[0, 1)` in code space before the floor (unbiased stochastic rounding for QAT), while `eval()` is the plain deterministic floor; both take the signed clamp `[-2^(adc_bits-1), 2^(adc_bits-1) - 1]`.
- **Ownership.** Fabricated state lives in the device children; the tile owns no static mismatch, so `_sample_fabricate_mismatch` is an explicit base no-op and the cascade fans into the children.

---

- **Reference**: [xbar base](../../../../reference/primitive/macro/cim/README.md)
- **Implementation**: `neurox/primitive/macro/cim/base.py`, `neurox/primitive/macro/cim/ideal.py`
- **Tests**: `tests/primitive/macro/test_ideal_cim_macro_rescale.py`
