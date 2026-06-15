# Xbar Abstract Layer

## Summary

Every physical crossbar — the offset-coded 1T1R tile, a future differential 1T1R tile, other topologies — implements one abstract contract: the same primitive operation, the same integer value domain, and the same output-rescale grid. This document specifies that contract and the lossless ideal twin that any physical xbar is compared against. Topology-specific physics lives in the family directories (e.g. [_1t1r/](_1t1r/README.md)).

## Primitive operation

A tile carries two operations:

- **program** — write a weight: a non-negative integer digit tensor; digits combine positionally with radix $r$.
- **VMM read** — drive an input activation onto the rows, settle to the DC operating point, and digitize the per-column accumulated result to a signed integer code at a chosen ADC operating point (mode and resolution $b$).

A row shares one input; a column aggregates into one output. The logical weight range (digit count, signing, column grouping) is decided above the tile by the [mapper](../mapper/README.md); the tile publishes only the integer value domain it can physically carry: the input grid $\mathcal{X}$, the digit count $D$, and the radix $r$.

## Output rescale

A read returns a signed integer code, not a current. The map between the code and the ideal integer dot product $M_{\text{ideal}}$ is the rescale factor $s$, with the signed convention $M_{\text{ideal}} \approx \text{code}\cdot s$. Each xbar owns its $(\text{mode}, b) \to s$ lookup: a physical tile reads $s$ from a calibrated table (see [calibration guide](../../guides/calibration/README.md)); the ideal twin derives $s$ from integer geometry alone.

## Ideal twin

The ideal twin (`IdealXbar`) is the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid, but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against, and ADC calibration uses the physical-vs-ideal gap to pick comparator thresholds.

Its rescale at resolution $b$ is derived from integer geometry alone,

$$s = \frac{M_{\max}}{2^{\,b-1}-1}, \qquad M_{\max} = N_{\mathrm{row}}\cdot \max|w_{\text{logical}}|\cdot \max|x|,$$

and its VMM is an exact integer dot product: collapse the digit axis with the radix-weighted vector $(1, r, r^2, \dots, r^{D-1})$ to form $M_{\text{ideal}}$, then quantize,

$$\text{code} = \operatorname{clamp}\!\left(\left\lfloor \frac{M_{\text{ideal}}}{s} \right\rfloor,\ -2^{\,b-1},\ 2^{\,b-1}-1\right).$$

Coarsely, $M_{\text{ideal}} = \mathbf{w}_{\text{logical}}^{\!\top}\mathbf{x}$ exactly. The sentinel $b = 0$ returns $M_{\text{ideal}}$ unmodified; $b = 1$ is degenerate ($2^{0}-1 = 0$ divides by zero) and is unsupported. The ADC mode is opaque to the ideal twin.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathcal{X}$ | integer input grid a row can carry | — | `x_range` |
| $D$ | digits per tile-word | — | `w_digit_count` |
| $r$ | positional radix | — | `w_digit_radix` |
| $M_{\text{ideal}}$ | ideal integer dot product | — | — |
| $\mathbf{w}_{\text{logical}}, x$ | logical weight, activation (runtime inputs) | — | — |
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
