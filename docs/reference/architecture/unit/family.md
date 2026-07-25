# Unit family

A unit is an exact-integer drop-in replacement for one PyTorch operator — `F.linear` or `F.conv2d` — realized by decomposing a high-precision quantized-integer matrix multiply onto the native integer value domain of the execution substrate and recombining the partial reads into one integer result. The family contract factors out what every member shares: the operator law, the matmul-shaped lowering template, the integer-bias domain, the two orthogonal placement axes — matrix tiling and precision slicing — and the radix-weighted shift-add that inverts the slicing.

## Operator law

Unit operators are exact-integer replicas of `F.linear` / `F.conv2d`: over its accepted integer value domain a unit computes the same result as the replaced PyTorch function applied to the same integer operands, returned pre-requantize; the only deviation is the substrate's own non-ideality (for CIM, the per-tile ADC quantization of each constituent read). There is no public matmul operator — a matmul consumer is a linear consumer with `bias=None` (the engine-internal primitive keeps the `matmul` name).

**Lowering template.** Every operator lowers onto one protected matmul-shaped substrate call through three hook seams: a program-time weight-to-matrix map, a call-time activation-to-planes map, and a call-time aggregation-undo that removes exactly the axes the second seam introduced. `F.linear` uses the identity weight map and a size-1 plane axis; `F.conv2d` uses geometry-parameterized seams ([conv2d](conv2d.md)).

**Shape-deviation law.** Whoever introduces a shape change deviating from the replaced function's expectation undoes it: unit-introduced value slicing is recombined by radix-weighted shift-add, unit-introduced tiling and padding are accumulated and trimmed, and unit-introduced operator lowering axes are folded back. Caller-owned axes — batch dims, any time axis — ride through untouched: the unit never reduces, reorders, or interprets a leading dim.

**Integer bias domain.** The bias belonging to `F.linear` / `F.conv2d` semantics lives inside the unit: it is programmed alongside the weight as an integer vector and added in the int64 accumulation domain, before any requantization. Bias preloads the final full-scale accumulation stage, after the last shift-add; it costs zero additional cycles and zero dynamic energy, and its static area/leakage belongs to the unit-level PPA fields.

**Linear lowering.** With trailing contraction axis $K$ and programmed weight $\mathbf{W}$ of shape $(N, K)$,

$$y_{\ldots,n} = \sum_{k} x_{\ldots,k}\, W_{n,k} + b_n,$$

realized by inserting a size-1 plane axis, running the substrate matmul, removing it, and adding the bias $b$ (if programmed).

## Shared conventions

A unit is value-domain only: it accepts integer weights and activations within its published value ranges and returns an integer, pre-requantize result. Requantization back to the activation grid lies outside its scope; the only bias it adds is the integer bias of the operator law above.

Two orthogonal axes place the matmul on physical tiles: a matrix-**tiling** axis ($T_r$, $T_c$) that splits any matmul too large for one tile, and a precision-**slicing** axis ($S_w$, $S_a$; specific to compute-in-memory) that decomposes a high-precision value into tile-carriable pieces. Matrix tiling is application-neutral — it applies to any matmul and adds no value decomposition. The slice counts $S_w$, $S_a$ are config-given, not inferred; the degenerate $S_w = S_a = 1$ performs no slicing.

Precision slicing is LSB-first. A value whose range exceeds one macro's
`w_value_range` or `x_value_range` is decomposed into positional slices, and
every slice is itself a complete logical value accepted by one macro call.
The unit never observes how the macro encodes that value internally. The
per-slice positional weight is the slice radix $R$, and the LSB-first slice
weights are $(1, R, R^{2}, \dots)$.

Decompose and aggregate are inverse operations: slicing a value into positional slices and the radix-weighted shift-add that recombines the per-tile partial reads are dual.

## Governing laws

Over its accepted value domain a unit computes an integer dot product matching exact matrix multiplication. For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$, both restricted to the unit's integer value ranges,

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

returned as a pre-requantize integer tensor. This integer $\mathbf{Y}$ is the pre-ADC ideal the decomposition reconstructs exactly; the realized result carries only the per-tile ADC quantization of each constituent read.

**Matrix tiling.** A weight matrix wider or taller than one macro is split into
a grid: $T_r = \lceil N / N_{\mathrm{out}} \rceil$ along the output axis and
$T_c = \lceil K / N_{\mathrm{in}} \rceil$ along the contraction axis, where
`output_num` and `input_num` are the macro's logical port capacities. The reads
on the $T_c$ contraction tiles are summed by plain integer accumulation.

**Slice recombination.** A value recombines from its slices $m_i$ by the radix-weighted shift-add

$$M = \sum_{i} m_i\, R^{i},$$

the aggregation primitive that folds the slice axis with the positional weights $(1, R, R^2, \dots)$, specified in [digital/shift_adder](../../primitive/digital/shift_adder.md).

**Output rescale.** For any ADC operating point a unit publishes a recovery-side rescale factor $s$ relating the integer dot product to the digitized code,

$$M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s.$$

A unit exposes a discrete set of operating points and a maximum resolution across them, both inherited from the tiles it aggregates; the rescale convention and its calibration are the [physical-tile contract](../../primitive/macro/cim/family.md#governing-laws). A degenerate member performing an exact integer computation carries no output quantization: it exposes a single operating point with unit rescale, $s = 1$.

## Noise & non-idealities

A unit adds no non-ideality of its own: the lowering, slicing, and aggregation arithmetic is exact by construction. Every deviation from the exact integer result enters through the tiles it aggregates — their analog non-idealities and ADC quantization — specified in the [physical-tile contract](../../primitive/macro/cim/family.md) and the topology families beneath it.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathbf{W}$ | logical weight matrix (runtime input) | — | `weight` |
| $\mathbf{X}$ | logical activation matrix (runtime input) | — | `input` |
| $\mathbf{Y}$ | pre-requantize integer output | — | `linear` / `conv2d` return |
| $N, K, M$ | output, contraction, and activation-row dims | — | `w_logical_shape`, input shape |
| $b$ | integer bias vector (length $N$ or $C_{\mathrm{out}}$) | — | `int_bias` |
| $S_w, S_a$ | weight-, activation-slice counts (precision-slicing axis) | — | `w_slice_num`, `x_slice_num` |
| $T_r, T_c$ | output-, contraction-axis tile counts (matrix-tiling axis) | — | — |
| $N_{\mathrm{in}}, N_{\mathrm{out}}$ | macro logical input / output capacity | — | `input_num`, `output_num` |
| $R$ | positional radix between adjacent slices | — | `slice_radix` |
| $m_i$ | value carried by slice $i$ | — | — |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |

## Assumptions, scope & validity

Stated assumptions:

- A unit returns a pre-requantize integer result; requantization lies outside its scope, and the only bias it adds is the integer bias of the operator law.
- The contracts are defined for integer weights and activations within the value ranges the unit accepts.
- The decomposition is value-domain exact: the only deviation from the exact integer result is the analog non-ideality of the constituent tile reads, not the lowering, slicing, or aggregation arithmetic.

TODO (domain author): state the validity boundary of the slice-and-shift-add
decomposition, the largest dot-product magnitude before ADC clipping, and any
regime where the value-domain-exact assumption breaks.

## References

TODO.

---

- **Internals**: [unit base internals](../../../internals/architecture/unit/base.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../api/README.md)
