# Conv2dUnit / IdealConv2dUnit

`neurox/architecture/unit/conv2d.py` defines the `Conv2dUnit` operator ABC, the `F.conv2d` specialization of the [UnitBase](base.md) template. `neurox/architecture/unit/ideal/conv2d.py` defines the substrate-free exact-integer `IdealConv2dUnit` reference.

## Design decisions

- **Conv overrides the template with geometry-parameterized seams.** `Conv2dUnit.conv2d` is the concrete template: `_conv2d_out_hw` → `_conv2d_planes` (conv seam 2) → `_matmul` → `_conv2d_fold` (conv seam 3) → the per-channel bias add. The fold needs the per-call `(H_out, W_out)`, so `out_hw` is passed explicitly through both conv seams — no per-call state is ever stashed on `self`.
- **The rank is canonicalized once, at the face.** `conv2d` rejects any rank other than 3-D or 4-D, unsqueezes a 3-D input to `B = 1`, and squeezes the batch axis back off the result — so the caller keeps `F.conv2d` parity while everything below the face is written against strictly `[B, C_in, H, W]`. Both seams and the geometry helper therefore state one concrete shape chain instead of a leading `...`.
- **Geometry storage is one shared init helper.** `_init_conv2d_operator` stores kernel size, stride, padding, and dilation on protected attrs and initializes the `_int_bias` slot; a concrete host calls it once in its constructor. `_conv2d_out_hw` computes the output map from these attrs and raises on a non-positive extent.
- **The interface does not import implementations.** The operator ABC is independent of `CimUnit`; the ideal leaf depends on both interfaces from `architecture/unit/ideal/conv2d.py`. It takes stride, padding, and dilation from its config while the kernel extent comes from the 4-D `w_logical_shape`.
- **The ideal conv reference is int64-only.** `_conv2d_planes` is a digital im2col by precomputed integer index grids and advanced indexing, not `F.unfold` (whose kernels are float-oriented): dtype-agnostic data movement, exact for any integer dtype. `_matmul` contracts in int64 with no fp32 fast path; `quantization_mode, adc_bits` are accepted and unused (lossless reference). `_weight_to_matrix` flattens the kernel to `[C_out, C_in*kh*kw]` at program time; `_conv2d_fold` transposes and unflattens `L` back to `(H_out, W_out)`.

## Contracts & invariants

- **`conv2d` matches `F.conv2d` shape semantics.** `[B, C_in, H, W]` maps to `[B, C_out, H_out, W_out]`, and a 3-D `[C_in, H, W]` input returns 3-D — the same two ranks `F.conv2d` accepts, and no other. Grouped convolution is unsupported. The bias adds as `_int_bias.view(-1, 1, 1)` over the output map, after the fold, in the int64 accumulation domain.
- **Conv seams undo exactly their own axes.** `_conv2d_planes` introduces the plane axes its host lowers with; `_conv2d_fold` removes exactly those axes (LIFO axis-stack discipline, [UnitBase](base.md)).
- **`IdealConv2dUnit.program` gates shape.** Any shape other than the 4-D `w_logical_shape` raises `ValueError`; the ideal reference does not enforce the CIM-backed integer boundary. The `(C_out,)` bias goes through `_program_int_bias`.
- **Zero-padding needs a representable zero.** A host configured with non-zero padding must require its `x_value_range` to cover 0 — zero-padding injects `x = 0` activations. (The ideal leaf computes in int64 regardless; its config ranges are reported surface.)
- **Sentinel quantization surface.** `adc_max_bits is None` and `rescale_factor == 1.0` — same lossless sentinel as [IdealLinearUnit](linear.md).
- **No substrate, hence zero duration.** `latency__ns(input_shape, *, adc_bits)` returns `0.0` and reads no window count out of the shape: the reference holds neither macro nor engine schedule, so no time axis exists below it.

## Gotchas

- **Reference only.** The ideal conv leaf is a value-domain baseline with zero unit-local PPA — never an energy baseline or a production accuracy result. It is the bit-exact oracle for the engine-backed lowering.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md), [conv2d mapping](../../../reference/architecture/unit/conv2d.md)
- **Implementation**: `neurox/architecture/unit/conv2d.py`, `neurox/architecture/unit/ideal/conv2d.py`
- **Tests**: `tests/architecture/unit/test_conv2d_cim_unit.py`
