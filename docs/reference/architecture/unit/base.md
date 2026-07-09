# Unit family

A unit decomposes a high-precision quantized-integer matrix multiply onto the native integer value domain of one or more crossbar tiles and recombines the per-tile partial reads into one integer dot product. It factors out the architecture every member shares: the two orthogonal axes — matrix tiling and precision slicing — that place a matmul on the tiles, and the radix-weighted shift-add that inverts the slicing.

## Shared conventions

A unit is value-domain only: it accepts integer weights and activations within its published value ranges and returns an integer, pre-requantize dot product. Bias addition and the requantization back to the activation grid lie outside its scope.

Two orthogonal axes place the matmul on physical tiles: a matrix-**tiling** axis ($T_r$, $T_c$) that splits any matmul too large for one tile, and a precision-**slicing** axis ($S_w$, $S_a$; specific to compute-in-memory) that decomposes a high-precision value into tile-carriable pieces. Matrix tiling is application-neutral — it applies to any matmul and adds no value decomposition. The slice counts $S_w$, $S_a$ are config-given, not inferred; the degenerate $S_w = S_a = 1$ performs no slicing.

Precision slicing is LSB-first. A **value** — role-neutral: a weight on the weight side, an activation on the input side — whose range exceeds what one tile cell can carry is decomposed into positional **slices**, each a fixed-capacity piece of $D$ **digits** (the integer symbol one xbar cell carries at digit radix $r$). The per-slice positional weight is the **slice radix** $R = r^{D}$, and the LSB-first slice weights are $(1, R, R^{2}, \dots)$. The per-slice value range follows from the digit count $D$ and the digit radix $r$ published by the [physical-tile contract](../../primitive/macro/cim/README.md), the authority for the digit/slice interface.

Decompose and aggregate are inverse operations: slicing a value into positional slices and the radix-weighted shift-add that recombines the per-tile partial reads are dual.

## Governing laws

Over its accepted value domain a unit computes an integer dot product matching exact matrix multiplication. For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$, both restricted to the unit's integer value ranges,

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

returned as a pre-requantize integer tensor. This integer $\mathbf{Y}$ is the pre-ADC ideal the decomposition reconstructs exactly; the realized result carries only the per-tile ADC quantization of each constituent read.

**Matrix tiling.** A weight matrix wider or taller than one tile is split into a grid of tiles: $T_r = \lceil N / N_{\mathrm{col}} \rceil$ along the output axis (rows of the transposed weight) and $T_c = \lceil K / N_{\mathrm{row}} \rceil$ along the contraction axis. The reads on the $T_c$ contraction tiles are summed back into one dot product by plain integer accumulation.

**Slice recombination.** A value recombines from its slices $m_i$ by the radix-weighted shift-add

$$M = \sum_{i} m_i\, R^{i},$$

the aggregation primitive that folds the slice axis with the positional weights $(1, R, R^2, \dots)$, specified in [digital/shift_adder](../../primitive/digital/shift_adder.md).

**Output rescale.** For any ADC operating point a unit publishes a recovery-side rescale factor $s$ relating the integer dot product to the digitized code,

$$M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s.$$

A unit exposes a discrete set of operating points and a maximum resolution across them, both inherited from the tiles it aggregates; the rescale convention and its calibration are the [physical-tile contract](../../primitive/macro/cim/README.md#output-rescale). A degenerate member performing an exact integer matmul carries no output quantization: it exposes a single operating point with unit rescale, $s = 1$.

## Noise & non-idealities

A unit adds no non-ideality of its own: the slicing and aggregation arithmetic is exact by construction. Every deviation from the exact integer dot product enters through the tiles it aggregates — their analog non-idealities and ADC quantization — specified in the [physical-tile contract](../../primitive/macro/cim/README.md) and the topology families beneath it.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathbf{W}$ | logical weight matrix (runtime input) | — | `weight` |
| $\mathbf{X}$ | logical activation matrix (runtime input) | — | `input` |
| $\mathbf{Y}$ | pre-requantize integer output | — | `matmul` return |
| $N, K, M$ | output, contraction, and activation-row dims | — | `w_logical_shape`, input shape |
| $S_w, S_a$ | weight-, activation-slice counts (precision-slicing axis) | — | `w_slice_num`, `x_slice_num` |
| $T_r, T_c$ | output-, contraction-axis tile counts (matrix-tiling axis) | — | — |
| $D$ | digits per slice (from the tile) | — | `w_digit_count` |
| $r$ | digit radix (from the tile) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | `slice_radix` |
| $m_i$ | value carried by slice $i$ | — | — |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |

## Assumptions, scope & validity

Stated assumptions:

- A unit returns a pre-requantize integer result; bias and requantization lie outside its scope.
- The decomposition is defined for integer weights and activations within the value ranges the unit accepts.
- The decomposition is value-domain exact: the only deviation from the exact integer dot product is the analog non-ideality of the constituent tile reads, not the slicing or aggregation arithmetic.

TODO (domain author): state the validity boundary of the slice-and-shift-add decomposition — the exact per-slice value range per encoding, the saturation of the positional recombination $M = \sum_i m_i R^i$, the largest dot-product magnitude representable before the ADC code clamps, and any regime where the value-domain-exact assumption breaks.

## References

TODO.

---

- **Internals**: [unit base internals](../../../internals/architecture/unit/base.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../api/README.md)
