# CIM macro family

## Shared conventions

A CIM macro carries two primitive operations:

- **Program** writes a logical integer weight matrix.
- **VMM read** accepts one logical input vector and returns one code per output at a
  selected quantization operating point — a mode and a resolution.

For $N$ inputs and $M$ outputs, `program` accepts
`[*inst_shape, N, M]`, `vec_mat_mul` accepts `[..., N]`, and the result has
shape `[..., M]`. The logical interface does not expose whether inputs or
weights are internally split into digits, how those digits map to physical
rows or columns, or how partial analog results are combined.

The owner supplies `input_num` and `output_num` when constructing a macro.
Physical implementations immediately map those values to their own geometry,
typically `row_num` and `col_num`, while the ideal implementation retains the
logical names. The abstract family does not publish geometry properties; upper
layers retain the logical capacities they supplied.

One conversion selects at most $A$ input positions. The caller forces all
unselected positions to zero; a selected position still counts against the
limit when its data value is zero. The selection limit sets the maximum
conversion dot-product magnitude and therefore the ADC calibration range.

A quantization mode is one declared conversion window over the exact integer
dot: an inclusive MAC-unit range $[M_{\min}, M_{\max}]$ holding
$N_{\mathrm{q}} = M_{\max} - M_{\min} + 1$ targets, canonical in the sense that
it is either unsigned, $M_{\min} = 0$, or mid-zero, $[-m, m-1]$. Those two are
the only shapes placing the zero point on a bin edge at every resolution, so
the zero code follows from the shape rather than from a separate
calibration. Each mode
declares one window; every resolution the mode supports reads the same window
and the same references in full, because resolution is realized inside the
converter, so lowering $b$ never changes sign recovery or zero-point semantics.

A mode is a name the macro passes to its reference source, which returns that
mode's reference values. The converter itself is mode-blind: it receives an
analog input, the references, and a resolution, and never a mode identity.

`x_value_range` and `w_value_range` expose the inclusive envelopes accepted by
the logical ports. A concrete encoding may leave holes inside an envelope.

## Governing laws

A read returns one code per output. The rescale factor $s$ is the currency
converting one macro's code into the code the ideal macro of the same windows
would return:

$$\mathrm{code}_{\mathrm{ideal}} \approx \mathrm{code}\cdot s.$$

The ideal macro is its own reference, so its factor is the identity $s=1$ at
its maximum resolution; a physical implementation obtains its maximum-resolution
factor $s_B$ from calibration, one per mode. Resolution enters through one law
shared by every implementation,

$$s_b = s_B\,2^{\,B-b},$$

because dropping a bit doubles the dot-product span one code carries. Recovering
MAC units from an ideal code is the consumer's step: it multiplies by the
window's quantization step $N_{\mathrm{q}}/2^{b}$, which the macro layer never
applies.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid | — | `x_value_range` |
| $\mathcal{W}$ | integer weight grid envelope | — | `w_value_range` |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $M_{\min}, M_{\max}$ | inclusive conversion window of one quantization mode | — | `quantization_input_ranges` |
| $N_{\mathrm{q}}$ | quantization targets held by a window | — | — |
| $s$ | output rescale factor | — | `rescale_factor` |
| $b, B$ | ADC resolution and its maximum | — | `adc_bits`, `adc_max_bits` |
| $N$ | logical input count supplied at construction | — | `input_num` |
| $M$ | logical output count supplied at construction | — | `output_num` |
| $A$ | maximum selected inputs per conversion | — | `max_active_num` |

## Assumptions, scope & validity

Programmed weights and input values conform to the concrete macro's logical
encoding, and every conversion selects at most $A$ input positions.

TODO (domain author): state the calibrated validity range of the output-rescale
relation.
