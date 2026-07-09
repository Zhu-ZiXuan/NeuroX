# CimMacro family

Every physical crossbar — any coding scheme on any cell topology — shares one primitive operation, one integer value domain, and one output-rescale grid, and is compared against a lossless ideal twin that preserves all three.

## Primitive operation

A tile carries two operations:

- **program** — write a weight as an integer digit tensor; digits combine positionally with radix $r$.
- **VMM read** — drive an input activation onto the rows and digitize the per-column accumulated result to a signed integer code at an ADC operating point (a mode and resolution $b$).

A row shares one input; a column aggregates into one output. The tile carries a role-neutral integer value, asserting no weight-or-activation role; the value domain it can physically carry is fixed by the input grid $\mathcal{X}$, the digit count $D$, and the digit radix $r$, from which the slice radix $R = r^{D}$ follows.

## Output rescale

A read returns a signed integer code. The rescale factor $s$ maps that code back to the ideal integer dot product $M_{\mathrm{ideal}}$ under the signed convention $M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s$. The rescale is a function of the ADC operating point (mode and resolution $b$): for a physical tile $s$ is obtained by calibration (see [calibration guide](../../../../guides/calibration/README.md)), while the ideal twin derives $s$ analytically from integer geometry.

## Ideal twin

The ideal twin is the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid, but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against. A faithful twin inherits the physical tile's geometry and ADC operating points, so it is directly comparable to that tile.

Its rescale at resolution $b$ follows from integer geometry alone,

$$s = \frac{M_{\max}}{2^{\,b-1}-1}, \qquad M_{\max} = N_{\mathrm{row}}\cdot \max|v|\cdot \max|x|,$$

This is the binary / uniform-quantization form, which maps $M_{\max}$ onto the top symmetric code $2^{\,b-1}-1$ and so requires $b \geq 2$; at $b = 1$ the denominator $2^{\,b-1}-1 = 0$ is undefined and the rescale is degenerate.

Its VMM is an exact integer dot product: collapse the digit axis with the radix-weighted vector $(1, r, r^2, \dots, r^{D-1})$ to form $M_{\mathrm{ideal}}$, then quantize. The read quantizer is a deterministic floor,

$$\mathrm{code} = \operatorname{clamp}\!\left(\left\lfloor \frac{M_{\mathrm{ideal}}}{s} \right\rfloor,\ -2^{\,b-1},\ 2^{\,b-1}-1\right).$$

Here $M_{\mathrm{ideal}} = \mathbf{v}^{\!\top}\mathbf{x}$ holds exactly; this integer $M_{\mathrm{ideal}}$ is the pre-ADC ideal, while a realized read carries the per-tile ADC quantization of the $\operatorname{clamp}/\lfloor\cdot\rfloor$ above. In the lossless limit — no ADC quantization — the read returns $M_{\mathrm{ideal}}$ exactly.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid a row can carry | — | `x_range` |
| $D$ | digits per slice (digit count) | — | `w_digit_count` |
| $r$ | digit radix (base of one cell's digit) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | — |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |
| $\mathbf{v}, \mathbf{x}$ | logical value vector and input (activation) vector the tile carries (runtime inputs) | — | — |
| $M_{\max}$ | maximum dot-product magnitude | — | `_max_dot_abs` |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $N_{\mathrm{row}}$ | number of rows | — | `row_num` |

## Assumptions, scope & validity

TODO (domain author): the validity of the integer-exact ideal model (digit-decomposition saturation, value-range limits) and the range of the rescale convention.

## Validation

TODO: link [validation/xbar](../../../../validation/README.md) — physical-vs-ideal code agreement under a noise-off policy.

## References

TODO.

---

- **Internals**: [CimMacro base internals](../../../../internals/primitive/macro/cim/base.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: `CimMacroConfig`, `IdealCimMacroConfig` (see `api`)
