# Xbar Abstract Layer

## Summary

Every physical crossbar — the offset-coded 1T1R tile, a future differential 1T1R tile, other topologies — implements one abstract contract: the same primitive operation, the same integer value domain, and the same output-rescale grid. This document specifies that contract and the lossless ideal twin that any physical xbar is compared against. Topology-specific physics lives in the family directories (e.g. [_1t1r/](_1t1r/README.md)).

## Primitive operation

A tile carries two operations:

- **program** — write a weight: a non-negative integer digit tensor; digits combine positionally with radix $r$.
- **VMM read** — drive an input activation onto the rows, settle to the DC operating point, and digitize the per-column accumulated result to a signed integer code at a chosen ADC operating point (mode and resolution $b$).

A row shares one input; a column aggregates into one output. The xbar is the authority for the digit/slice interface: it publishes the digit count per slice $D$, the digit radix $r$, the slice radix $R = r^{D}$, and the per-slice value range computed from $D$ and $r$. The weight- and activation-slice counts are config-given, not inferred. The tile carries a role-neutral integer value; it asserts no weight or activation role and exposes only the integer value domain it can physically carry: the input grid $\mathcal{X}$, the digit count $D$, and the radix $r$.

## Output rescale

A read returns a signed integer code, not a current. The map between the code and the ideal integer dot product $M_{\mathrm{ideal}}$ is the rescale factor $s$, with the signed convention $M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s$. Each xbar owns its $(\mathrm{mode}, b) \to s$ lookup: a physical tile reads $s$ from a calibrated table (see [calibration guide](../../guides/calibration/README.md)); the ideal twin derives $s$ from integer geometry alone.

## Ideal twin

The ideal twin (`IdealXbar`) is the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid, but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against, and ADC calibration uses the physical-vs-ideal gap to pick comparator thresholds. The faithful twin is derived from a fabricated physical xbar via `to_ideal()`, which keeps its geometry and ADC surface bound to the device; building one directly from a standalone config is a bring-up / reference convenience only and yields a synthetic, uncalibrated reference — not a production hardware accuracy or PPA result.

Its rescale at resolution $b$ is derived from integer geometry alone,

$$s = \frac{M_{\max}}{2^{\,b-1}-1}, \qquad M_{\max} = N_{\mathrm{row}}\cdot \max|v|\cdot \max|x|,$$

This is the binary / uniform-quantization form, which maps $M_{\max}$ onto the top symmetric code $2^{\,b-1}-1$ and so requires $b \geq 2$; at $b = 1$ the denominator $2^{\,b-1}-1 = 0$ is undefined and the rescale is degenerate. This matches the ADC operating-point minimum of $b = 2$.

Its VMM is an exact integer dot product: collapse the digit axis with the radix-weighted vector $(1, r, r^2, \dots, r^{D-1})$ to form $M_{\mathrm{ideal}}$, then quantize,

$$\mathrm{code} = \operatorname{clamp}\!\left(\left\lfloor \frac{M_{\mathrm{ideal}}}{s} \right\rfloor,\ -2^{\,b-1},\ 2^{\,b-1}-1\right).$$

Informally, $M_{\mathrm{ideal}} = \mathbf{v}^{\!\top}\mathbf{x}$ precisely; this integer $M_{\mathrm{ideal}}$ is the pre-ADC ideal, while a realized read carries the per-tile ADC quantization of the $\operatorname{clamp}/\lfloor\cdot\rfloor$ above. The sentinel $b = 0$ returns $M_{\mathrm{ideal}}$ unmodified — the no-noise, lossless limit with ADC quantization bypassed; $b = 1$ is degenerate ($2^{0}-1 = 0$ divides by zero) and is unsupported. The ADC mode is opaque to the ideal twin.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid a row can carry | — | `x_range` |
| $D$ | digits per slice (digit count) | — | `w_digit_count` |
| $r$ | digit radix (base of one cell's digit) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | `slice_radix` |
| $\mathcal{V}$ | per-slice value range, computed from $D$ and $r$ | — | `value_range` |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |
| $\mathbf{v}, \mathbf{x}$ | value vector and input value grid the tile carries (runtime inputs) | — | — |
| $M_{\max}$ | maximum dot-product magnitude | — | `max_dot_abs` |
| $s$ | output rescale factor | — | `rescale_factor` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $N_{\mathrm{row}}$ | number of rows | — | `row_num` |

## Assumptions, scope & validity

TODO (domain author): the validity of the integer-exact ideal model (digit-decomposition saturation, value-range limits) and the range of the rescale convention.

## Validation

TODO: link [validation/xbar](../../validation/README.md) — physical-vs-ideal code agreement under a noise-off policy.

## References

TODO.

---

- **Internals**: [xbar base internals](../../internals/xbar/base.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
