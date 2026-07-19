# IdealCimUnit

The degenerate `CimUnit` family member: it joins the registry without owning a tile, stores the integer weight, and runs `torch.matmul` against it.

## Design decisions

- **In the registry, but skips every layered step.** It registers via `@CimUnit.register_key(IdealCimUnitConfig)` so it shares the family factory and the `QuantMatMul` surface, but it declares no `xbar`, builds none (`_build_cim_macro` is never called), carries an empty `IdealCimUnitPolicy`, and sets its `IdealCimUnitConfig` unit-local peripheral PPA (`area_per_inst__um2` / `leakage_per_inst__uW`) to zero — the no-overhead reference. The family signature args `policy` / `dtype` / `T__K` / `ideal_xbar` are accepted for uniformity and ignored.
- **Sentinel ADC surface.** `adc_mode_num == 1`, `adc_max_bits == 0`, `adc_rescale_factor == 1.0`. The `0` bit count is the "no output quantization" sentinel — see [matmul](../matmul.md).
- **0-d nominal weight buffer.** `weight` and `nominal_weight` register as 0-d `int32` buffers (`persistent=False`), giving `weight` a defined attribute placeholder before any `program` call while keeping it out of the state dict.

## Contracts & invariants

- **`program` shape gate, then verbatim store.** `program(weight)` rejects any shape other than `w_logical_shape`, then stores the tensor unchanged into `self.weight` — no encoding, no slicing.
- **`matmul` is exact on both arithmetic paths.** `__init__` sizes the dot bound `K * max|x| * max|w|` from the config value ranges: below `2^24` the contraction runs as an fp32 `torch.matmul` — every product and partial sum stays exactly representable in IEEE fp32 (framework-default matmul precision, TF32 disabled) and the cast back to `int64` is lossless — which is what makes the unit GPU-capable, since CUDA has no integer-matmul kernel. At or above the bound both operands widen to `int64` before `torch.matmul`, exact at any magnitude but CPU-by-design. Fast-path exactness assumes range-conformant operands.

## Performance & resources

A single dense matmul (fp32 on the fast path, `int64` otherwise); no tiling, no analog cost. Its own peripheral PPA is the reference zero and it owns no circuit children to report any, so profiler aggregation over an ideal unit is sparse.

## Gotchas

- **Not a stand-in for a physical mode in a PPA study.** Use it only as a value-domain reference, not an energy baseline.
- **Bring-up / reference only, never a production accuracy result.** Directly instantiating an ideal member (this degenerate unit, or a unit wrapping a directly built `IdealCimMacro`) yields a synthetic, uncalibrated reference; a hardware-faithful ideal twin comes from `to_ideal()` on a physical config.

---

- **Reference**: [ideal](../../../../reference/architecture/unit/cim/ideal.md)
- **Implementation**: `neurox/architecture/unit/cim/ideal.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_ideal_cim_unit_fp32_exact.py`
