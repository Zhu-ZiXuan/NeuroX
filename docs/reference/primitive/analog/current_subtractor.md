# Current subtractor

## Physical model

A single-ended current-mode subtractor: it forms the difference of two leg currents and emits that difference as a non-negative magnitude current together with a one-bit direction (sign). The magnitude path steers the two legs to a high and a low rail and outputs their gained difference, so the delivered current is unipolar; an independent comparator resolves which leg was larger and drives the sign bit. Two departures from the exact difference are modelled: a static ratio mismatch on the subtracted leg (the gain error's physical source) and an input-referred current offset on the comparison (the sign decision's error source). The block is a pure current-domain primitive: it draws no rail energy of its own — the downstream current-domain consumer that owns the rail tallies dissipation.

## Governing equations

The block forms the offset, ratio-scaled signed difference of the two legs,

$$\Delta I = I_a - \rho \, I_b + I_{\mathrm{os}},$$

then emits its gained magnitude and direction bit,

$$I_{\mathrm{diff}} = g \, \lvert \Delta I \rvert, \qquad s = [\, \Delta I < 0 \,],$$

with $g$ the nominal current gain, $\rho$ the subtracted-leg ratio (nominally 1), $I_{\mathrm{os}}$ the input-referred offset, $[\,\cdot\,]$ the Iverson bracket (1 where the predicate holds), and $s$ set where the subtracted $I_b$ leg dominates.

## Numerical method

N/A — closed form, no iteration.

## Noise & non-idealities

Two sources, both static per-instance Gaussians fixed at fabrication and independent per element:

- **Ratio mismatch** — the subtracted leg carries a multiplicative (unit-mean) Gaussian ratio $\rho = 1 + \mathcal{N}(0, \sigma_\rho)$ with relative sigma $\sigma_\rho$; it is the physical source of the subtraction's gain error. Off leaves the exact unit ratio.
- **Sign-comparator offset** — an additive input-referred current offset $I_{\mathrm{os}} = \mathcal{N}(0, \sigma_{\mathrm{os}})$ [uA] shifts the signed difference before the sign decision; it can flip $s$ near balance, where the magnitude is smallest. Off leaves a zero offset.

The shared Gaussian mismatch law is in [nonideality](../nonideality.md).

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `gain` ($g$) | nominal current gain of the subtraction stage | — | $> 0$ | Design |
| `mismatch_sigma_relative` ($\sigma_\rho$) | relative subtracted-leg ratio-mismatch sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| `offset_sigma__uA` ($\sigma_{\mathrm{os}}$) | sign-comparator input-referred offset sigma, fixed at fabrication | uA | $\geq 0$ | Measured |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_a$ | added (first) leg current | uA | `i_a__uA` |
| $I_b$ | subtracted (second) leg current | uA | `i_b__uA` |
| $\Delta I$ | offset, ratio-scaled signed difference | uA | intermediate |
| $I_{\mathrm{diff}}$ | output magnitude current | uA | `subtract` return |
| $s$ | direction (sign) bit | — | `subtract` return |
| $g$ | nominal current gain | — | `gain` |
| $\rho$ | subtracted-leg ratio (nominally 1) | — | `ratio_mismatch` |
| $\sigma_\rho$ | relative ratio-mismatch sigma | — | `mismatch_sigma_relative` |
| $I_{\mathrm{os}}$ | sign-comparator input-referred offset | uA | `offset__uA` |
| $\sigma_{\mathrm{os}}$ | offset sigma | uA | `offset_sigma__uA` |

## Assumptions, scope & validity

The magnitude output is unipolar (non-negative) by construction; the signed information travels in the separate direction bit. The subtraction stays linear only while the subtractor operates inside its (unmodelled) compliance band; the model assumes this condition holds. The offset is input-referred to the difference node, so it perturbs the magnitude and the sign identically — the sign flip near balance is its dominant observable effect.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the current-subtractor and sign-comparator models.

---

- **Internals**: [current_subtractor internals](../../../internals/primitive/analog/current_subtractor.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `CurrentSubtractorConfig`, `CurrentSubtractorPolicy` (see `api`)
