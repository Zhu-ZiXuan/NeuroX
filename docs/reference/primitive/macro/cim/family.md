# CIM macro family

## Shared conventions

A CIM macro carries two primitive operations:

- **Program** writes a logical integer weight matrix.
- **VMM read** accepts one logical input vector and returns one code per output at a
  selected ADC operating point.

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

`x_value_range` and `w_value_range` expose the inclusive envelopes accepted by
the logical ports. A concrete encoding may leave holes inside an envelope.

## Governing laws

A read returns one code per output. The rescale factor $s$ maps that code to an
ideal integer conversion result $M_p$:

$$M_p \approx \mathrm{code}\cdot s.$$

The rescale is indexed by ADC mode and resolution. A physical implementation
obtains it from calibration.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid | — | `x_value_range` |
| $\mathcal{W}$ | integer weight grid envelope | — | `w_value_range` |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $b$ | ADC resolution | — | `adc_bits` |
| $N$ | logical input count supplied at construction | — | `input_num` |
| $M$ | logical output count supplied at construction | — | `output_num` |
| $A$ | maximum selected inputs per conversion | — | `max_active_num` |

## Assumptions, scope & validity

Programmed weights and input values conform to the concrete macro's logical
encoding, and every conversion selects at most $A$ input positions.

TODO (domain author): state the calibrated validity range of the output-rescale
relation.

---

- **Internals**: [CIM macro base](../../../../internals/primitive/macro/cim/base.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `CimMacroConfig` (see `api`)
