# Current mirror

## Physical model

An ideal single-ended current copy: the output branch current is an exact ratio scaling of the input branch current. The copy carries no finite output impedance (no Early-effect current droop), no headroom or compliance limit, and no input-referred offset. The one modelled departure from this lossless ideal is a static copy-ratio mismatch. The block is a pure current-copy primitive: it draws no rail energy of its own — the downstream current-domain consumer that owns the rail tallies dissipation.

## Governing equations

The output branch current is the ratio-scaled copy of the input branch current,

$$I_{\mathrm{out}} = r \, I_{\mathrm{in}},$$

with $r$ the dimensionless mirror ratio.

## Numerical method

N/A — closed form, no iteration.

## Noise & non-idealities

One source: copy-ratio mismatch — a static per-instance multiplicative (Pelgrom) Gaussian $r (1 + \mathcal{N}(0, \sigma_r))$ fixed at fabrication, with relative sigma $\sigma_r$, independent per mirror element. The shared area-scaled mismatch law is in [nonideality](../nonideality.md).

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `mirror_ratio` ($r$) | output/input copy ratio | — | $> 0$ | Design |
| `ratio_sigma_relative` ($\sigma_r$) | relative copy-ratio mismatch sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | input branch current | uA | `i_in__uA` |
| $I_{\mathrm{out}}$ | output branch current | uA | `replicate` return |
| $r$ | output/input copy ratio | — | `mirror_ratio` |
| $\sigma_r$ | relative copy-ratio mismatch sigma | — | `ratio_sigma_relative` |

## Assumptions, scope & validity

The mirror is modelled as a ratio-copy element whose only departure from a lossless ideal is the copy-ratio mismatch; it is a pure current-copy primitive that accounts no rail energy of its own. The copy stays linear only while the mirror operates inside its (unmodelled) compliance band; the model assumes this condition holds.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the current-mirror copy model.

---

- **Internals**: [current_mirror internals](../../internals/analog/current_mirror.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `CurrentMirrorConfig`, `CurrentMirrorPolicy` (see `api`)
