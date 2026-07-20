# Conv2dUnit / IdealConv2dUnit

`neurox/architecture/unit/conv2d.py`: the `Conv2dUnit` operator ABC — the `F.conv2d` specialization of the [UnitBase](base.md) template — and the concrete `IdealConv2dUnit`, the substrate-free exact-integer conv2d reference member of the `CimUnit` registry.

## Design decisions

- **Conv overrides the template with geometry-parameterized seams.** `Conv2dUnit.conv2d` is the concrete template: `_conv2d_out_hw` → `_conv2d_planes` (conv seam 2) → `_matmul` → `_conv2d_fold` (conv seam 3) → the per-channel bias add. The fold needs the per-call `(H_out, W_out)`, so `out_hw` is passed explicitly through both conv seams — no per-call state is ever stashed on `self`.
- **Geometry storage is one shared init helper.** `_init_conv2d_operator` stores kernel size, stride, padding, and dilation on protected attrs and registers the `int_bias` slot; a concrete host calls it once in its constructor. `_conv2d_out_hw` computes the output map from these attrs and raises on a non-positive extent.
- **The ideal leaf lives beside its operator ABC.** `IdealConv2dUnit{,Config,Policy}` are defined in the same module after a deferred `cim.base` import (same cycle-breaking pattern as [linear](linear.md)); the leaf registers via `@CimUnit.register_key(IdealConv2dUnitConfig)` and takes the conv geometry from its own config (`stride` / `padding` / `dilation`), while the kernel extent comes from the 4-D `w_logical_shape` — never a duplicate config field.
- **The ideal conv reference is int64-only.** `_conv2d_planes` is a digital im2col by precomputed integer index grids and advanced indexing, not `F.unfold` (whose kernels are float-oriented): dtype-agnostic data movement, exact for any integer dtype. `_matmul` contracts in int64 with no fp32 fast path; `adc_operation_point` is accepted and unused (lossless reference). `_weight_to_matrix` flattens the kernel to `[C_out, C_in*kh*kw]` at program time; `_conv2d_fold` transposes and unflattens `L` back to `(H_out, W_out)`.

## Contracts & invariants

- **`conv2d` matches `F.conv2d` shape semantics.** Trailing `[C_in, H, W]` maps to trailing `[C_out, H_out, W_out]`; every leading dim is a broadcast batch dim that rides through untouched. Grouped convolution is unsupported. The bias adds as `int_bias.view(-1, 1, 1)` over the output map, after the fold, in the int64 accumulation domain.
- **Conv seams undo exactly their own axes.** `_conv2d_planes` introduces the plane axes its host lowers with; `_conv2d_fold` removes exactly those axes (LIFO axis-stack discipline, [UnitBase](base.md)).
- **`IdealConv2dUnit.program` gates shape and dtype.** Any shape other than the 4-D `w_logical_shape` raises `ValueError`; floating or complex weight dtypes raise `TypeError`; the `(C_out,)` bias goes through `_program_int_bias`.
- **Zero-padding needs a representable zero.** A host configured with non-zero padding must require its `x_value_range` to cover 0 — zero-padding injects `x = 0` activations. (The ideal leaf computes in int64 regardless; its config ranges are reported surface.)
- **Sentinel ADC surface.** `adc_mode_num == 1`, `adc_max_bits == 0`, `adc_rescale_factor == 1.0` — same lossless sentinel as [IdealLinearUnit](linear.md).

## Gotchas

- **Reference only.** The ideal conv leaf is a value-domain baseline with zero unit-local PPA — never an energy baseline or a production accuracy result. The Toeplitz lowering of the physical conv unit is a different mapping entirely; this leaf is the oracle it is validated against.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md), [conv2d mapping](../../../reference/architecture/unit/conv2d.md)
- **Implementation**: `neurox/architecture/unit/conv2d.py`
- **Tests**: `tests/architecture/unit/test_conv2d_cim_unit.py`
