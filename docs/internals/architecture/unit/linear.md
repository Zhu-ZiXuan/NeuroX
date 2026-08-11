# LinearUnit / IdealLinearUnit

`neurox/architecture/unit/linear.py` defines the `LinearUnit` operator ABC, the `F.linear` specialization of the [UnitBase](base.md) template. `neurox/architecture/unit/ideal/linear.py` defines the substrate-free exact-integer `IdealLinearUnit` reference.

## Design decisions

- **Linear is the generic-seam operator.** `LinearUnit` overrides seams 2 and 3 only: `_activation_to_planes` inserts the size-1 `M` axis (`[..., K] -> [..., 1, K]`), `_undo_aggregation` removes it. `linear()` runs the inherited `_lower_matmul` template and adds the programmed integer bias in the int64 accumulation domain; `program(weight, bias=None)` stays abstract (the host owns the substrate write).
- **The interface does not import implementations.** The operator ABC is independent of `CimUnit`; the ideal leaf depends on both interfaces from `architecture/unit/ideal/linear.py`. Package initialization imports concrete leaves to establish registry membership without a deferred module-tail import.
- **Registry membership without a tile.** `IdealLinearUnit` registers its `(IdealLinearUnitConfig, IdealLinearUnitPolicy)` pair, so `CimUnitConfig.from_file` + `CimUnit.from_config` dispatch to it exactly like any engine-backed member; it owns no engine and no `xbar`, and `dtype` / `T__K` / `ideal_macro` are accepted for uniformity and ignored.
- **Sentinel quantization surface.** `adc_max_bits is None` — the "no output quantization" sentinel ([UnitBase](base.md)) — and `rescale_factor` is `1.0` for any argument pair.
- **No substrate, hence zero duration.** `latency__ns(input_shape, *, adc_bits)` returns `0.0`: the reference holds neither macro nor engine schedule, so no time axis exists below it and the operand layout says nothing about a duration.
- **Weight is program-produced state.** Construction allocates no nominal or placeholder weight. `program` stores the caller's tensor as an ordinary attribute, so execution requires programming and device migration must precede it.

## Contracts & invariants

- **`linear` matches `F.linear` shape semantics.** Trailing `[K]` contracts to trailing `[N]`; every leading dim (including a caller time axis) is a broadcast batch dim that rides through untouched. The substrate `_matmul` never includes the bias; only `linear` adds it.
- **`IdealLinearUnit.program` shape-gates, then stores verbatim.** Any shape other than `w_logical_shape` raises; the weight tensor is stored unchanged (no encoding, no slicing) and the `(N,)` bias goes through `_program_int_bias`.
- **`_matmul` is exact on both arithmetic paths.** `__init__` sizes the dot bound `K * max|x| * max|w|` from the config value ranges: below `2^24` the contraction runs as an fp32 `torch.matmul` — every product and partial sum stays exactly representable in IEEE fp32 (framework-default matmul precision, TF32 disabled) and the cast back to `int64` is lossless — which is what keeps the unit GPU-capable, since CUDA has no integer-matmul kernel. At or above the bound both operands widen to `int64` before `torch.matmul`, exact at any magnitude but CPU-by-design. Fast-path exactness assumes range-conformant operands.

## Gotchas

- **Not a stand-in for a physical mode in a PPA study.** The ideal leaf is a value-domain / lossless upper-bound reference with zero unit-local PPA and no circuit children; use it for flow bring-up and to isolate QAT issues from analog modelling, never as an energy baseline or a production accuracy result. A hardware-faithful ideal twin comes from `ideal_macro=True` on a physical config, not from this leaf.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/linear.py`, `neurox/architecture/unit/ideal/linear.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_ideal_cim_unit_fp32_exact.py`
