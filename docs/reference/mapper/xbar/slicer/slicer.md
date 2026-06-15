# Slice Decomposition

## Summary / role

A slicer is the value-decomposition step of the mapping: it takes a high-precision integer-valued tensor and produces a digit tensor whose two trailing axes are `[slice_num, digit_num]` — `slice_num` positional slices, each holding `digit_num` signed digits sized to an xbar's primitive cell. It sits one layer above the [transcoder](../../transcoder/README.md): the transcoder gives the single-integer encoding math, and the slicer applies it across a tensor and regroups the resulting digit string into the slice/digit lattice a tile consumes. The slicer is encoding-agnostic about tile *shape*: geometric tiling (padding, chunking the tensor across physical tiles) is decided above it. Two concrete decompositions exist — a serial path for activations and a direct digitise-then-group path for weights.

## Physical model

There is no physical object; the slicer is the integer-decomposition math that bridges a logical value to the digit lattice a physical cell grid can carry. The output's slice axis is the positional axis a downstream shift-add reducer recombines, and the digit axis is the per-slice signed-digit string the tile's cells hold in parallel.

The two decompositions correspond to the two data roles:

- **serial decomposition** (activation path) — one unsigned scalar becomes `slice_num` positional digits at radix $r$, each digit driving one input cycle of the tile. The digit axis is a structural singleton (`digit_num = 1`): an activation cell carries one digit per cycle. The activation grid is unsigned, so the encoding is fixed to true-form (sign-magnitude); a signed alphabet would emit negative digits that cannot fit the unsigned primitive cell.
- **direct decomposition** (weight path) — one scalar is encoded into a single digit string of length `slice_num * digit_count` and the trailing axis is regrouped into `[slice_num, digit_count]`. Consecutive `digit_count` digits form one xbar-word slice, least-significant first. The encoding is configurable (true-form, radix-complement, or canonical).

## Governing equations

Let $r$ be the per-digit radix and $D_{\mathrm{c}}$ (`digit_count`) the digits packed into one slice. The **slice radix** $R$ — the positional weight ratio between adjacent slices — and the LSB-first **slice weights** are

$$R = r^{\,D_{\mathrm{c}}}, \qquad \text{slice weights} = \big(1,\ R,\ R^{2},\ \dots,\ R^{\,S-1}\big),$$

where $S$ (`slice_num`) is the number of slices. The slice weights name how the slice axis recombines: a consumer reduces the slices by the shift-add

$$M = \sum_{i=0}^{S-1} m_i\, R^{i},$$

with $m_i$ the value carried by slice $i$. For the serial decomposition the digit count is the structural singleton $D_{\mathrm{c}} = 1$, so $R = r$ and the slice weights are the plain radix powers $(1, r, r^2, \dots)$.

The **value range** is the inclusive integer band one input scalar can take, given by the underlying encoding over the *total* digit count $S \cdot D_{\mathrm{c}}$:

- serial decomposition: unsigned positional, $[\,0,\ r^{S} - 1\,]$;
- direct decomposition: the encoding's value range (see [transcoder encodings](../../transcoder/encodings.md#governing-equations)) evaluated at $D = S \cdot D_{\mathrm{c}}$.

The output shapes are `slice(x).shape == x.shape + (slice_num, 1)` for the serial path and `x.shape + (slice_num, digit_count)` for the direct path.

### value_range vs digit_range

These two ranges live at different layers and must not be conflated:

- **value range** — the algorithm-side inclusive range of one *complete* input scalar across all its slices and digits. It is what the slicer exposes, and it answers "what logical values round-trip through this decomposition".
- **digit range** — the primitive single-*cell* range. It answers "what a single physical cell can hold".

The slicer's value range is the composed band over $S \cdot D_{\mathrm{c}}$ digits; the digit range is the per-position single-cell alphabet. Keeping the names distinct keeps the algorithm-side complete-value range separate from the primitive-cell range.

## Numerical method

The decomposition is exact integer arithmetic delegated to the [transcoder](../../transcoder/README.md): encode the scalar(s) into a flat digit string, then reshape the trailing axis into the slice/digit lattice. There is no iteration or floating-point error. The serial path encodes `slice_num` true-form digits and inserts the singleton digit axis; the direct path encodes `slice_num * digit_count` digits in one pass and unflattens the trailing axis into `[slice_num, digit_count]`, least-significant slice first.

## Noise & non-idealities

N/A - the slicer is exact integer arithmetic and introduces no noise. Quantization and analog error enter downstream in the device/circuit/ADC models.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `slice_num` | number of positional slices $S$ ($\ge 1$) | — | Design |
| `digit_count` | digits packed per slice $D_{\mathrm{c}}$ ($\ge 1$; direct path) | — | Design |
| `digit_radix` | per-digit positional radix $r$ ($\ge 2$) | — | Design |
| `encoding` | signed-digit policy (direct path; serial path fixed to true-form) | — | Design |

Provenance terms: [parameter_provenance](../../../parameter_provenance.md). The runtime integer tensor being sliced is an input, not a parameter. A slicer is built from the xbar's primitive-cell fields (the digit count, digit radix, and input grid) forwarded to the subclass; that forwarding is an implementation concern (see internals). Schema: `[mapper]` config (config reference).

## Assumptions, scope & validity

- Inputs are integer-valued and must lie within the slicer's value range to round-trip; out-of-range inputs wrap or truncate via the underlying encoding (no clamp).
- The serial (activation) path is unsigned by construction; its encoding is fixed to true-form because a signed alphabet would emit digits the unsigned primitive cell cannot carry.
- The slicer decides value decomposition only; geometric tiling of the tensor across physical tiles (padding, chunking) is not its responsibility.
- Digit-grid compatibility between the slicer output and the xbar's primitive cell is a caller-side invariant the slicer does not re-validate.

TODO (domain author): state any dtype/overflow validity limit on the largest $S \cdot D_{\mathrm{c}}$ for a given radix.

## Validation

TODO - link the slice/round-trip and shape-contract evidence (`tests/test_slicer.py`) once a `validation/mapper` page exists.

## References

TODO (domain author): cite the bit/digit-slicing and shift-add weighted-recombination scheme for compute-in-memory tiles.

---

- **Internals**: [slicer internals](../../../../internals/mapper/xbar/slicer/slicer.md)
- **Validation**: TODO - `validation/mapper` (not yet written)
- **Configuration**: [config reference](../../../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
