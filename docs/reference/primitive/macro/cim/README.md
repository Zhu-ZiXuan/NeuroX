# CimMacro family

Every physical crossbar — any coding scheme on any cell topology — shares one primitive operation, one integer value domain, and one output-rescale grid, and is compared against a lossless ideal twin that preserves all three.

## Primitive operation

A tile carries two operations:

- **program** — write a weight as an integer digit tensor; digits combine positionally with radix $r$.
- **VMM read** — drive an input activation onto the rows as serial row active phases and digitize each phase's per-column partial result to a signed integer code at an ADC operating point (a mode and resolution $b$).

A row shares one input; a column aggregates into one output. The tile carries a role-neutral integer value, asserting no weight-or-activation role; the value domain it can physically carry is fixed by the input grid $\mathcal{X}$, the digit count $D$, and the digit radix $r$, from which the slice radix $R = r^{D}$ follows.

## Row active phases

A VMM read activates $A$ rows at a time (`active_row_num`), so one read is a serial sequence of $P = N_{\mathrm{row}} / A$ **row active phases** (`active_phase_num`); phase $p$ drives rows $[pA, (p+1)A)$ while every other word line holds its off level. $A$ sets the analog dot-product dynamic range of one conversion, and hence the ADC calibration. Each phase is digitized independently, so a read returns **per-phase codes** with trailing shape $[P, N_{\mathrm{col}}]$ (column innermost, active-phase axis immediately left); the phase axis is always present — size 1 when $A = N_{\mathrm{row}}$. The macro boundary is digital → DAC → analog → ADC → digital, encapsulating the minimal analog chain; any accumulation or routing of per-phase codes is the caller's digital-domain decision, made at the unit layer. The row active phase is the serial row-activation step of one VMM; it is distinct from a readout-stage phase (a scheme's sub-phase timing within one conversion, e.g. a `t_phase` parameter).

## Output rescale

A read returns signed integer per-phase codes. The rescale factor $s$ maps a code back to the ideal integer partial dot product $M_{p}$ of its phase under the signed convention $M_{p} \approx \mathrm{code}\cdot s$. The rescale is a function of the ADC operating point (mode and resolution $b$): for a physical tile $s$ is obtained by calibration (see [calibration guide](../../../../guides/calibration/README.md)), while the ideal twin derives $s$ analytically from integer geometry.

## Ideal twin

The ideal twin is the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid, but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against. A faithful twin inherits the physical tile's geometry and ADC operating points, so it is directly comparable to that tile.

Its rescale at resolution $b$ follows from integer geometry alone. One ADC conversion digitizes one active phase, so the range is the per-phase partial dot,

$$s = \frac{M_{\max}}{2^{\,b-1}-1}, \qquad M_{\max} = A\cdot \max|v|\cdot \max|x|,$$

This is the binary / uniform-quantization form, which maps $M_{\max}$ onto the top symmetric code $2^{\,b-1}-1$ and so requires $b \geq 2$; at $b = 1$ the denominator $2^{\,b-1}-1 = 0$ is undefined and the rescale is degenerate.

Its VMM is an exact integer dot product per phase: collapse the digit axis with the radix-weighted vector $(1, r, r^2, \dots, r^{D-1})$, reduce each phase's own row block to the partial dot $M_{p}$, then quantize each phase. The read quantizer is a deterministic floor,

$$\mathrm{code}_{p} = \operatorname{clamp}\!\left(\left\lfloor \frac{M_{p}}{s} \right\rfloor,\ -2^{\,b-1},\ 2^{\,b-1}-1\right).$$

Here $\sum_{p} M_{p} = \mathbf{v}^{\!\top}\mathbf{x} = M_{\mathrm{ideal}}$ holds exactly; the integer partial dots $M_{p}$ are the pre-ADC ideal, while a realized read carries the per-phase ADC quantization of the $\operatorname{clamp}/\lfloor\cdot\rfloor$ above. Quantize-then-accumulate is the modeled physical semantics: in general $\sum_{p} Q(M_{p}) \neq Q\!\left(\sum_{p} M_{p}\right)$. In the lossless limit — no ADC quantization — the read returns the exact partial dots, whose sum over the phase axis is $M_{\mathrm{ideal}}$.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid a row can carry | — | `x_range` |
| $D$ | digits per slice (digit count) | — | `w_digit_count` |
| $r$ | digit radix (base of one cell's digit) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | — |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |
| $M_{p}$ | ideal integer partial dot product of active phase $p$ | — | — |
| $\mathbf{v}, \mathbf{x}$ | logical value vector and input (activation) vector the tile carries (runtime inputs) | — | — |
| $M_{\max}$ | maximum per-phase partial-dot magnitude | — | `_max_phase_dot_abs` |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $N_{\mathrm{row}}$ | number of rows | — | `row_num` |
| $N_{\mathrm{col}}$ | number of columns | — | `col_num` |
| $A$ | rows simultaneously activated per active phase | — | `active_row_num` |
| $P$ | serial active phases per VMM, $P = N_{\mathrm{row}} / A$ | — | `active_phase_num` |

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
