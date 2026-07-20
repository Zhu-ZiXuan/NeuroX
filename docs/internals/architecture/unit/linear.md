# LinearUnit / IdealLinearUnit

`neurox/architecture/unit/linear.py`: the `LinearUnit` operator ABC — the `F.linear` specialization of the [UnitBase](base.md) template — and the concrete `IdealLinearUnit`, the substrate-free exact-integer reference member of the `CimUnit` registry.

## Design decisions

- **Linear is the generic-seam operator.** `LinearUnit` overrides seams 2 and 3 only: `_activation_to_planes` inserts the size-1 `M` axis (`[..., K] -> [..., 1, K]`), `_undo_aggregation` removes it. `linear()` runs the inherited `_lower_matmul` template and adds the programmed integer bias in the int64 accumulation domain; `program(weight, bias=None)` stays abstract (the host owns the substrate write).
- **The ideal leaf lives beside its operator ABC.** `IdealLinearUnit{,Config,Policy}` are defined in the same module, after a deferred `from neurox.architecture.unit.cim.base import ...` (module-tail import: loading `cim.base` executes the `cim` package `__init__`, whose leaves import `LinearUnit` from this module — the ordering breaks the cycle). Cross-module imports in the unit tree name concrete modules, never a package `__init__`.
- **Registry membership without a tile.** `IdealLinearUnit` registers via `@CimUnit.register_key(IdealLinearUnitConfig)`, so `CimUnitConfig.from_file` + `CimUnit.from_config` dispatch to it exactly like any engine-backed member; it owns no engine and no `xbar`, carries an empty policy marker, and `dtype` / `T__K` / `ideal_xbar` are accepted for uniformity and ignored.
- **Sentinel ADC surface.** `adc_mode_num == 1`, `adc_max_bits == 0`, `adc_rescale_factor == 1.0`. The `0` bit count is the "no output quantization" sentinel ([UnitBase](base.md)).
- **0-d nominal weight buffer.** `weight` and `nominal_weight` register as 0-d `int32` buffers (`persistent=False`), giving `weight` a defined attribute placeholder before any `program` call while keeping it out of the state dict.

## Contracts & invariants

- **`linear` matches `F.linear` shape semantics.** Trailing `[K]` contracts to trailing `[N]`; every leading dim (including a caller time axis) is a broadcast batch dim that rides through untouched. The substrate `_matmul` never includes the bias; only `linear` adds it.
- **`IdealLinearUnit.program` shape-gates, then stores verbatim.** Any shape other than `w_logical_shape` raises; the weight tensor is stored unchanged (no encoding, no slicing) and the `(N,)` bias goes through `_program_int_bias`.
- **`_matmul` is exact on both arithmetic paths.** `__init__` sizes the dot bound `K * max|x| * max|w|` from the config value ranges: below `2^24` the contraction runs as an fp32 `torch.matmul` — every product and partial sum stays exactly representable in IEEE fp32 (framework-default matmul precision, TF32 disabled) and the cast back to `int64` is lossless — which is what keeps the unit GPU-capable, since CUDA has no integer-matmul kernel. At or above the bound both operands widen to `int64` before `torch.matmul`, exact at any magnitude but CPU-by-design. Fast-path exactness assumes range-conformant operands.

## Gotchas

- **Not a stand-in for a physical mode in a PPA study.** The ideal leaf is a value-domain / lossless upper-bound reference with zero unit-local PPA and no circuit children; use it for flow bring-up and to isolate QAT issues from analog modelling, never as an energy baseline or a production accuracy result. A hardware-faithful ideal twin comes from `ideal_xbar=True` on a physical config, not from this leaf.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/linear.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_ideal_cim_unit_fp32_exact.py`
