# Direct engine

The no-slice corner of the precision-slicing axis: one weight slice and one activation slice ($S_w = S_a = 1$). A weight value maps onto one tile's native value range unsliced through the weight codec, and an input value is already on the tile's per-cycle input grid — the activation slicing step is the identity member of the slicer family and decomposes nothing. The variant applies when the quantization grid already fits inside one tile's value range; binary $\{0, 1\}$ inputs are simply the caller's choice of a narrow integer alphabet.

## Physical model

The variant places each logical weight directly on one macro input/output
coordinate. A larger matrix is tiled as
$T_r=\lceil N/N_{\mathrm{out}}\rceil$ by
$T_c=\lceil K/N_{\mathrm{in}}\rceil$. The final tiles are zero-padded. There is
no precision-slice axis and no shift-add.

## Governing equations

The variant realizes the exact integer dot product

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k}$$

as a sum over the contraction-tile grid. The engine creates zero-masked input
phases, the macro converts each phase independently, the phase accumulator
combines those codes, and a plain sum across $T_c$ completes the contraction.
Output tiles concatenate and trim to $N$. `program` passes logical values
directly to the macro; any physical digit encoding is macro-internal.

## Noise & non-idealities

N/A at the variant level — the variant adds no non-ideality. ADC quantization and analog non-idealities enter through the constituent tile reads, specified by the [CIM macro family](../../../../primitive/macro/cim/family.md); the cross-tile accumulation is exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `cim_macro_config` | owned physical-tile configuration | — | — | Design |
| `phase_accumulator_config` | input-phase accumulator per macro output | — | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | — | Design |

Activations are unsigned true-form by definition, so the variant carries no activation encoding. Provenance terms: [module_parameter](../../../../../conventions/module_parameter.md); file-level schema: [config reference](../../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $N_{\mathrm{in}}$ | macro logical input capacity | — | `input_num` |
| $N_{\mathrm{out}}$ | macro logical output capacity | — | `output_num` |
| $T_r$ | output-axis tile count | — | structure count |
| $T_c$ | contraction-axis tile count | — | structure count |

The logical dims ($N$, $K$, $M$), the value-domain symbols, and the ADC surface are in [unit/family](../../family.md#symbols).

## Assumptions, scope & validity

- Weight and input values must fit the macro's published logical ranges. These
  owner-side contracts are not enforced by elementwise runtime scans.
- The variant applies only when the quantization grid fits one tile's value range; for $S_w > 1$ or $S_a > 1$ use inter_array_slice or intra_array_slice.

TODO (domain author): the exact value-range bound per encoding and the saturation behaviour at the tile boundary.

## Validation

Bit-exact parity against an int64 CPU matmul oracle — `tests/architecture/unit/test_linear_cim_unit.py`.

## References

TODO.

---

- **Internals**: [direct engine internals](../../../../../internals/architecture/unit/cim/engine/direct.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md)
