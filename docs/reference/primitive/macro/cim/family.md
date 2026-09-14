# CIM macro family

## Shared conventions

A CIM macro carries two primitive operations:

- **Program** writes a logical integer weight matrix.
- **VMM read** accepts one logical input vector and returns one code per output at a selected quantization operating point and optional ADC resolution.

For $N$ inputs and $M$ outputs, `program` accepts `[*inst_shape, N, M]`, `vec_mat_mul` accepts `[..., N]`, and the result has shape `[..., M]`. The logical interface does not expose whether inputs or weights are internally split into digits, how those digits map to physical rows or columns, or how partial analog results are combined.

The macro configuration fixes the logical input count $N$, parallel readout-lane count $L$, and serial scan count $S$. The logical output count is derived as

$$M = L S,$$

Each concrete macro owns its internal readout layout and restores results to the logical output order used by `program`. The public result always has shape `[..., M]`; its ordering is independent of the physical lane and scan assignment.

A read may select a prefix of the logical outputs. `effective_output_num` supplies one count per operation, without an output axis; its shape broadcasts to the instance-aligned leading shape of the input without enlarging it. A scalar applies the same count everywhere. The concrete output mapping determines which readout lanes participate in each scan. Each lane uses a contiguous scan prefix; operation duration is the longest lane's active scan count multiplied by the duration of one scan. The first selected number of logical outputs carries valid codes; every remaining position is zero-filled.

Disabled word-line selections and closed column branches change the electrical drive conditions. Currents that remain under those conditions retain their conduction cost. Conversion and control events are charged only when executed.

One conversion selects at most $A$ input positions. Unselected positions are zero; a selected position still counts against the limit when its data value is zero. The selection limit bounds the conversion dot-product magnitude used when designing and calibrating the readout.

A quantization mode is an integer index pairing one reference operating point with one calibrated macro output scale. The macro passes the index to its reference source and uses the factor at the same tuple position. The converter remains mode-blind: it receives an analog input, references, and a resolution rather than a mode identity.

The macro returns one of two numerical code mappings. Zero-point mapping quantizes into a centered two's-complement code and arithmetically truncates it to the active resolution. Sign-magnitude mapping quantizes the magnitude, truncates it, and restores the sign. Neither mapping is inferred from the logical input or weight ranges.

An explicit `adc_active_bits = b` requests a finite resolution in `[1, adc_bits]`. Passing `None` asks the receiving macro for its highest available precision: a physical macro uses its maximum ADC width, while an ideal macro returns the exact integer result without virtual quantization. The argument is required, so callers state this choice explicitly.

`x_value_range` and `w_value_range` expose the inclusive continuous integer ranges accepted by the logical ports. An encoding may use redundant digit strings but leaves no holes in either range.

## Governing laws

A read at finite active ADC resolution $b$ returns one final macro code per output. ADC-internal encoding and sign or zero-point processing remain inside the macro. Its effective rescale factor $s_b$ expresses one output code in MAC units:

$$M \approx \mathrm{code}_b\,s_b.$$

Each mode stores one calibrated factor $s_B$ at the maximum ADC resolution $B$. Other active resolutions derive their effective factor through

$$s_b = s_B\,2^{B-b},$$

because dropping one ADC decision bit doubles the full-resolution MAC span represented by one compact code.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $\mathcal{X}$ | integer input grid | — | `x_value_range` |
| $\mathcal{W}$ | integer weight grid | — | `w_value_range` |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $s_B$ | MAC units per output code at maximum ADC resolution | — | `rescale_factors` |
| $s_b$ | MAC units per output code at active resolution $b$ | — | `rescale_factor` |
| $b, B$ | active and maximum ADC resolution | — | `adc_active_bits`, `adc_bits` |
| $N$ | configured logical input capacity | — | `input_num` |
| $M$ | derived logical output capacity, $LS$ | — | `output_num` |
| $L$ | parallel readout-circuit groups | — | `lane_num` |
| $S$ | serial output positions per readout group | — | `scan_num` |
| $A$ | maximum selected inputs per conversion | — | `max_active_num` |

## Assumptions, scope & validity

Programmed weights and input values conform to the macro's logical encoding, and every conversion selects at most $A$ input positions.

TODO (domain author): state the calibrated validity range of the output-rescale relation.
