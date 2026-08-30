# CIM macro family

## Shared conventions

A CIM macro carries two primitive operations:

- **Program** writes a logical integer weight matrix.
- **VMM read** accepts one logical input vector and returns one code per output at a selected quantization operating point and ADC resolution.

For $N$ inputs and $M$ outputs, `program` accepts `[*inst_shape, N, M]`, `vec_mat_mul` accepts `[..., N]`, and the result has shape `[..., M]`. The logical interface does not expose whether inputs or weights are internally split into digits, how those digits map to physical rows or columns, or how partial analog results are combined.

Logical input and output capacities are fixed at construction. Internal physical geometry is not part of the family interface.

One conversion selects at most $A$ input positions. Unselected positions are zero; a selected position still counts against the limit when its data value is zero. The selection limit bounds the conversion dot-product magnitude used when designing and calibrating the readout.

A quantization mode is an integer index pairing one reference operating point with one calibrated macro output scale. The macro passes the index to its reference source and uses the factor at the same tuple position. The converter remains mode-blind: it receives an analog input, references, and a resolution rather than a mode identity.

The macro returns one of two numerical code mappings. Zero-point mapping quantizes into a centered two's-complement code and arithmetically truncates it to the active resolution. Sign-magnitude mapping quantizes the magnitude, truncates it, and restores the sign. Neither mapping is inferred from the logical input or weight ranges. Lossless ideal execution is a separate bypass rather than a third mapping.

`x_value_range` and `w_value_range` expose the inclusive envelopes accepted by the logical ports. A concrete encoding may leave holes inside an envelope.

## Governing laws

A read at active ADC resolution $b$ returns one final macro code per output. ADC-internal encoding and sign or zero-point processing remain inside the macro. Its effective rescale factor $s_b$ expresses one output code in MAC units:

$$M \approx \mathrm{code}_b\,s_b.$$

Each mode stores one calibrated factor $s_B$ at the maximum ADC resolution $B$. Other active resolutions derive their effective factor through

$$s_b = s_B\,2^{B-b},$$

because dropping one ADC decision bit doubles the full-resolution MAC span represented by one compact code.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid | — | `x_value_range` |
| $\mathcal{W}$ | integer weight grid envelope | — | `w_value_range` |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $s_B$ | MAC units per output code at maximum ADC resolution | — | `rescale_factors` |
| $s_b$ | MAC units per output code at active resolution $b$ | — | `rescale_factor` |
| $b, B$ | active and maximum ADC resolution | — | `adc_active_bits`, `adc_bits` |
| $N$ | logical input count supplied at construction | — | `input_num` |
| $M$ | logical output count supplied at construction | — | `output_num` |
| $A$ | maximum selected inputs per conversion | — | `max_active_num` |

## Assumptions, scope & validity

Programmed weights and input values conform to the macro's logical encoding, and every conversion selects at most $A$ input positions.

TODO (domain author): state the calibrated validity range of the output-rescale relation.
