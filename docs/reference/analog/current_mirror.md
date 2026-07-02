# Current mirror

## Summary / role

The `CurrentMirror` is a leaf analog block: a single-ended current-copy element that reproduces an input branch current at the output branch, scaled by a copy ratio. It is a structural building block shared across readout chains — one concrete block a consuming circuit composes directly, not a polymorphic family. The block performs the ratio copy and tallies the data-dependent rail energy; its one error hook is a copy-ratio mismatch toggle.

## Physical model

Physically, a current mirror senses the input branch current and forces a matched (ratio-scaled) copy through the output branch by tying the gate-source bias of the output device to that of the diode-connected input device. The input current is sourced externally — its production energy is accounted by the upstream block — so the block tallies only the output-branch draw from the rail supply for the duration of the read window.

The copy is exact at the configured ratio when no nonideality is enabled, with no finite output impedance (no Early-effect current droop), no headroom/compliance limit, and no input-referred offset. The one modelled departure besides the data-dependent rail dissipation is an optional copy-ratio mismatch (see Noise & non-idealities).

## Governing equations

The output branch current is the ratio-scaled copy of the input branch current,

$$I_{\mathrm{out}} = r \, I_{\mathrm{in}},$$

with $r$ the dimensionless mirror ratio (`mirror_ratio`). The rail dissipation over one read window counts the output branch only,

$$E_{\mathrm{dyn}} = V_{\mathrm{supply}} \, |I_{\mathrm{out}}| \, t_{\mathrm{read}},$$

with $V_{\mathrm{supply}}$ the rail voltage (`v_supply__V`) and $t_{\mathrm{read}}$ the read-window width (`read_pulse__ns`) passed by the owning xbar. The input current is sourced externally and its production energy is accounted by the upstream block, so it is not counted here. In the consistent unit set $\mathrm{uA} \times \mathrm{V} \times \mathrm{ns} = \mathrm{fJ}$.

## Numerical method

N/A — the copy is a closed-form per-call map; no iteration.

## Noise & non-idealities

One source: copy-ratio mismatch. When the `mismatch` toggle on `CurrentMirrorPolicy` is set, the copy ratio is perturbed by a multiplicative (Pelgrom) Gaussian $r (1 + \mathcal{N}(0, \sigma_r))$ with relative sigma `ratio_sigma_relative`, sampled per call element-wise (not a fixed per-instance offset). With the toggle off, the copy is the exact ratio. No finite output impedance, headroom/compliance limit, or input offset is modelled.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `mirror_ratio` ($r$) | output/input copy ratio | — | Design |
| `v_supply__V` ($V_{\mathrm{supply}}$) | rail supply voltage driving the data-dependent dissipation | V | Design |
| `ratio_sigma_relative` ($\sigma_r$) | relative copy-ratio mismatch sigma, gated by the `mismatch` policy | — | Design |
| `read_pulse__ns` ($t_{\mathrm{read}}$) | read-window width scaling the per-call energy | ns | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the mirror is a ratio-copy element with an optional copy-ratio mismatch. No finite output impedance, headroom/compliance ceiling, or input offset is modelled; the modelled departures from a lossless ideal are the data-dependent rail dissipation on the output branch and the optional copy-ratio mismatch. The model is valid where the mirror operates inside its (unmodelled) compliance band so the copy stays linear.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the current-mirror copy and the rail-dissipation model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | input branch current | uA | `i_in__uA` |
| $I_{\mathrm{out}}$ | output branch current | uA | `replicate` return |
| $r$ | output/input copy ratio | — | `mirror_ratio` |
| $\sigma_r$ | relative copy-ratio mismatch sigma | — | `ratio_sigma_relative` |
| $V_{\mathrm{supply}}$ | rail supply voltage | V | `v_supply__V` |
| $t_{\mathrm{read}}$ | read-window width | ns | `read_pulse__ns` |
| $E_{\mathrm{dyn}}$ | per-call rail dissipation | fJ | logged dynamic energy |

---

- **Internals**: [current_mirror internals](../../internals/analog/current_mirror.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `CurrentMirrorConfig`, `CurrentMirrorPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
