# CimMacro family

Every physical crossbar — any coding scheme on any cell topology — shares one primitive operation, one integer value domain, and one output-rescale grid, and is compared against a lossless ideal twin that preserves all three.

## Primitive operation

A tile carries two operations:

- **program** — write a weight as an integer digit tensor; digits combine positionally with radix $r$.
- **VMM read** — one independent analog conversion per WL plane: drive the plane's codes onto the rows and digitize each column's partial result to a signed integer code at an ADC operating point (a mode and resolution $b$).

A row shares one input; a column aggregates into one output. The tile carries a role-neutral integer value, asserting no weight-or-activation role; the value domain it can physically carry is fixed by the input grid $\mathcal{X}$, the digit count $D$, and the digit radix $r$, from which the slice radix $R = r^{D}$ follows.

## Active-row window

One conversion activates at most $A$ rows (`active_row_num`, published to upper layers as the single query `max_active_rows`); every other word line holds its off level. $A$ sets the analog dot-product dynamic range of one conversion, and hence the ADC calibration. A full-row read is therefore a serial sequence of $N_{\mathrm{row}} / A$ WL planes, but the serialization belongs to the caller: the tile consumes WL planes with primitive trailing $[N_{\mathrm{row}}]$ — every leading axis is anonymous broadcast batch the tile never inspects, reorders, or reduces, and rows outside the caller's active window must arrive zeroed (WL off) — and returns codes with the same leading order and primitive trailing $[N_{\mathrm{col}}]$. The tile performs no phase expansion and no accumulation; the macro boundary is digital → DAC → analog → ADC → digital, encapsulating the minimal analog chain, and any routing or accumulation of leading axes is the caller's digital-domain decision, made at the unit layer (the engine's sub-phase axis). The active-row window is a row-activation constraint of one conversion; it is distinct from a readout-stage timing phase within one conversion (e.g. a `t_phase` parameter).

## Output rescale

A read returns one signed integer code per column per WL plane. The rescale factor $s$ maps a code back to the ideal integer plane dot product $M_{p}$ under the signed convention $M_{p} \approx \mathrm{code}\cdot s$. The rescale is a function of the ADC operating point (mode and resolution $b$): for a physical tile $s$ is obtained by calibration (see [calibration guide](../../../../guides/calibration/README.md)), while the ideal twin derives $s$ analytically from integer geometry.

## Ideal twin

The ideal twin is the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid, but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against. A faithful twin inherits the physical tile's geometry and ADC operating points, so it is directly comparable to that tile.

Its rescale at resolution $b$ follows from integer geometry alone. One ADC conversion digitizes one WL plane, and a conformant plane carries at most $A$ live rows, so the per-conversion range is

$$s = \frac{M_{\max}}{2^{\,b-1}-1}, \qquad M_{\max} = A\cdot \max|v|\cdot \max|x|,$$

This is the binary / uniform-quantization form, which maps $M_{\max}$ onto the top symmetric code $2^{\,b-1}-1$ and so requires $b \geq 2$; at $b = 1$ the denominator $2^{\,b-1}-1 = 0$ is undefined and the rescale is degenerate.

Its VMM is an exact integer dot product per WL plane: collapse the digit axis with the radix-weighted vector $(1, r, r^2, \dots, r^{D-1})$, contract each plane against the full row extent (zeroed rows contribute nothing) to the plane dot $M_{p}$, then quantize each plane independently. The read quantizer is a deterministic floor,

$$\mathrm{code}_{p} = \operatorname{clamp}\!\left(\left\lfloor \frac{M_{p}}{s} \right\rfloor,\ -2^{\,b-1},\ 2^{\,b-1}-1\right).$$

When the caller's planes partition the rows of one logical read, $\sum_{p} M_{p} = \mathbf{v}^{\!\top}\mathbf{x} = M_{\mathrm{ideal}}$ holds exactly; the integer plane dots $M_{p}$ are the pre-ADC ideal, while a realized read carries the per-plane ADC quantization of the $\operatorname{clamp}/\lfloor\cdot\rfloor$ above. Quantize-then-accumulate is the modeled physical semantics — in general $\sum_{p} Q(M_{p}) \neq Q\!\left(\sum_{p} M_{p}\right)$ — and it is preserved under the caller-owned serialization because each sub-phase arrives as its own plane. In the lossless limit — no ADC quantization — the read returns the exact plane dots, whose caller-side sum is $M_{\mathrm{ideal}}$.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid a row can carry | — | `x_range` |
| $D$ | digits per slice (digit count) | — | `w_digit_count` |
| $r$ | digit radix (base of one cell's digit) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | — |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |
| $M_{p}$ | ideal integer dot product of WL plane $p$ | — | — |
| $\mathbf{v}, \mathbf{x}$ | logical value vector and input (activation) vector the tile carries (runtime inputs) | — | — |
| $M_{\max}$ | maximum per-conversion plane-dot magnitude | — | `_max_plane_dot_abs` |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $N_{\mathrm{row}}$ | number of rows | — | `row_num` |
| $N_{\mathrm{col}}$ | number of columns | — | `col_num` |
| $A$ | maximum simultaneously active rows per conversion | — | `active_row_num` (config), `max_active_rows` (query) |

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
