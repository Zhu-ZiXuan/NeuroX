# UnitBase

`neurox/architecture/unit/base.py`: the root operator ABC — the value-range / ADC / `fabricate` surface every unit publishes, the protected matmul-shaped lowering template with its three hook seams, and the integer-bias slot — plus the shared `_validate_int_bias` helper.

## Design decisions

- **One template, three seams.** The lowering skeleton is `_lower_matmul`: seam 2 (`_activation_to_planes`) → the abstract substrate primitive `_matmul` (`[..., M, K] -> [..., M, N]`, leading order preserved) → seam 3 (`_undo_aggregation`). Seam 1 (`_weight_to_matrix`) maps the logical operator weight to the `(N, K)` matrix at program time. All three seams default to the identity, so an operator ABC overrides only what its lowering needs.
- **No public matmul operator.** Unit operators are exact-integer replicas of `F.linear` / `F.conv2d`; a matmul consumer is a linear consumer with `bias=None`. `engine.matmul` keeps its name as the internal primitive — `_matmul` on the unit is the protected substrate seam, never a consumer surface.
- **The contract carries no PPA surface.** The base fixes the value-domain surface a consumer calls and stops there: area and leakage are absent by design, not by omission. Silicon is accounted on the module tree the profiler walks, an axis orthogonal to this contract — so what a unit reports is settled there, never by conforming here.
- **`_int_bias` is programmed state, not a buffer.** The class-level default is `None`; `_program_int_bias` stores the validated int64 tensor as an ordinary attribute, and `bias=None` clears it. It therefore follows the project lifecycle: move the unit before programming, and reprogram after any later migration.
- **Bias validation is centralized.** `_validate_int_bias` rejects non-integer dtypes and any shape other than `(channels,)`, and returns the int64 cast — the accumulation domain.

## Contracts & invariants

- **Bias is an accumulation-register preload.** Bias preloads the final full-scale accumulation stage, after the last shift-add; it costs zero additional cycles and zero dynamic energy, and its static area/leakage belongs to the unit-level PPA fields. In code the operator layer adds it as an int64 term after the substrate returns — the mathematical image of the preload.
- **Seam 3 undoes exactly the axes seam 2 introduced (LIFO axis stack).** Each layer introduces its serial axis immediately left of the inst-aligned block and undoes exactly its own axis. Caller-owned leading dims — batch, any time axis — ride through untouched: the template never reduces, reorders, or interprets a leading dim.
- **`program` is bound to one `w_logical_shape`.** The shape is fixed at construction and `program` rejects any other; re-`program` overwrites, it does not re-shape. Each operator ABC declares its own `program(weight, bias=None)` signature and channel count for the bias.
- **The ADC surface is published, not enforced.** `adc_mode_num` / `adc_max_bits` advertise the operating-point range; `adc_rescale_factor` raises `KeyError` for an uncalibrated point. `adc_max_bits == 0` is the "no output quantization" sentinel — not a one-level ADC. Value ranges are published capability, not a base-enforced runtime bound.
- **`fabricate` is part of the lifecycle surface.** The base requires a no-argument resampling operation but does not prescribe how a host stores or traverses fabricated state.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/base.py`
- **Tests**: `tests/architecture/unit/test_linear_cim_unit.py`
