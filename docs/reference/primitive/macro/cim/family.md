# CIM macro family

## Shared conventions

A CIM macro carries two primitive operations:

- **Program** writes an integer digit tensor whose digits combine positionally
  with radix $r$.
- **VMM read** drives one WL plane and returns one integer code per column at a
  selected ADC operating point.

A row shares one input and a column aggregates one output. The value domain is
fixed by the input grid $\mathcal{X}$, digit count $D$, and digit radix $r$.
The corresponding slice radix is $R=r^D$.

One conversion activates at most $A$ rows. All other word lines remain at their
off level. The active-row limit sets the maximum plane-dot magnitude and
therefore the ADC calibration range.

## Governing laws

A read returns one signed code per column per WL plane. The rescale factor $s$
maps that code to an ideal integer plane dot $M_p$:

$$M_p \approx \mathrm{code}\cdot s.$$

The rescale is indexed by ADC mode and resolution. A physical implementation
obtains it from calibration.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid | — | `x_value_range` |
| $D$ | digits per programmed value | — | `w_digit_count` |
| $r$ | digit radix | — | `w_digit_radix` |
| $R$ | slice radix, $r^D$ | — | — |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $b$ | ADC resolution | — | `adc_bits` |
| $N_{\mathrm{row}}$ | row count | — | `row_num` |
| $N_{\mathrm{col}}$ | column count | — | `col_num` |
| $A$ | maximum active rows per conversion | — | `active_row_num` |

## Assumptions, scope & validity

The programmed digits and input values lie inside their published integer
ranges, and every read plane contains at most $A$ active rows.

TODO (domain author): state the saturation boundary of the digit decomposition
and the calibrated validity range of the output-rescale relation.

---

- **Internals**: [CIM macro base](../../../../internals/primitive/macro/cim/base.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `CimMacroConfig` (see `api`)
