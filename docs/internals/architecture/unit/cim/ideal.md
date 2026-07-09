# IdealCimUnit

The degenerate `CimUnit` family member: it joins the registry without owning a tile, stores the integer weight, and runs `torch.matmul` against it.

## Design decisions

- **In the registry, but skips every layered step.** It registers via `@CimUnit.register_key(IdealCimUnitConfig)` so it shares the family factory and the `QuantMatMul` surface, but it declares no `xbar`, builds none (`_build_cim_macro` is never called), and carries an empty `IdealCimUnitPolicy`. The family signature args `policy` / `dtype` / `T__K` / `ideal_xbar` are accepted for uniformity and ignored.
- **Sentinel ADC surface.** `adc_mode_num == 1`, `adc_max_bits == 0`, `adc_rescale_factor == 1.0`. The `0` bit count is the "no output quantization" sentinel — see [base](../base.md).
- **0-d nominal weight buffer.** `weight` and `nominal_weight` register as 0-d `int32` buffers (`persistent=False`), giving `weight` a defined attribute placeholder before any `program` call while keeping it out of the state dict.

## Contracts & invariants

- **`program` shape gate, then verbatim store.** `program(weight)` rejects any shape other than `w_logical_shape`, then stores the tensor unchanged into `self.weight` — no encoding, no slicing.
- **`matmul` widens to int64.** Both operands cast to `int64` before `torch.matmul` so the full-width integer contraction cannot overflow.

## Performance & resources

A single dense `int64` matmul; no tiling, no analog cost. No PPA contribution (no constituent circuits), so profiler aggregation over an ideal unit is sparse.

## Gotchas

- **Not a stand-in for a physical mode in a PPA study.** It has no children, so it contributes no area / leakage / energy; use it only as a value-domain reference, not an energy baseline.
- **Bring-up / reference only, never a production accuracy result.** Directly instantiating an ideal member (this degenerate unit, or a unit wrapping a directly built `IdealCimMacro`) yields a synthetic, uncalibrated reference; a hardware-faithful ideal twin comes from `to_ideal()` on a physical config.

---

- **Reference**: [ideal](../../../../reference/architecture/unit/cim/ideal.md)
- **Implementation**: `neurox/architecture/unit/cim/ideal.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
