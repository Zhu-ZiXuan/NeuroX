# Voltage mux

## Summary / role

The `VoltageMux` is a leaf differential voltage-transport block in the readout chain: it moves a grouped differential signal from one stage to the next, applying a per-leg transport gain (a matched gain with an optional static inter-leg mismatch) and optionally injecting transport noise. It only transports and tallies the transport energy. It is a single concrete block a consuming circuit composes directly, not a polymorphic family.

## Physical model

Physically, the mux transports the differential signal from one stage to the next through a switched path. Each leg's passive insertion gain is the on-resistance/load divider $g \approx R_{\mathrm{load}}/(R_{\mathrm{load}}+R_{\mathrm{on}}) \le 1$ (an illustrative origin, not the parameter's definition), so the transported pair is attenuated relative to its input rather than passed through unchanged.

The block models the transport as a per-leg gain plus two additive transport-noise terms, separating two physically distinct non-ideality families:

- **Static (deterministic, fixed per built device).** The two legs' on-resistances differ by a per-die process offset, so their insertion gains differ. This **inter-leg gain mismatch** is the dominant first-order static differential error of a differential transport stage: it is multiplicative and converts the transported common mode into a signal-dependent differential error (finite CMRR — see Noise & non-idealities). It is modelled by promoting the matched gain $g$ to per-leg gains $g^{\pm} = g(1 \pm \tfrac{1}{2}\varepsilon_g)$, with $\varepsilon_g$ drawn once at fabricate. Frequency response / finite settling, signal-dependent $R_{\mathrm{on}}(V)$ nonlinearity, and the charge-injection / clock-feedthrough pedestal are further static or switching-event effects deliberately neglected here (see Assumptions).
- **Dynamic (random, re-sampled every transport).** Thermal/sampling ($kT/C$) and coupling disturbances on the switched path. These are zero-mean and signal-independent, and are modelled by the two additive noise terms below. They do **not** model the static mismatch.

## Governing equations

For a differential input pair each leg is scaled by its own static insertion gain and perturbed by the two optional noise terms,

$$V_{\mathrm{out}}^{\pm} = g\,\Bigl(1 \pm \tfrac{1}{2}\varepsilon_g\Bigr)\,V_{\mathrm{in}}^{\pm} + n_{\mathrm{cm}} \pm n_{\mathrm{dm}},$$

where $g$ is the matched transport gain (`mux_gain`), $\varepsilon_g$ the fabricate-fixed fractional inter-leg gain mismatch (zero when the `mux_gain_mismatch` policy switch is off), $n_{\mathrm{cm}}$ the common-mode noise sample (shared by both legs) and $n_{\mathrm{dm}}$ the differential-mode noise sample (added to the positive leg, subtracted from the negative leg); each noise term is zero-mean and zero when its policy switch is off. Taking the leg difference,

$$V_{\mathrm{out}}^{+}-V_{\mathrm{out}}^{-} = g\,V_{\mathrm{dm}} + g\,\varepsilon_g\,V_{\mathrm{cm}} + 2\,n_{\mathrm{dm}},$$

with $V_{\mathrm{cm}} = \tfrac{1}{2}(V_{\mathrm{in}}^{+}+V_{\mathrm{in}}^{-})$ and $V_{\mathrm{dm}} = V_{\mathrm{in}}^{+}-V_{\mathrm{in}}^{-}$: the wanted differential gain $g\,V_{\mathrm{dm}}$, the static common-mode-to-differential leakage $g\,\varepsilon_g\,V_{\mathrm{cm}}$, and the dynamic differential noise. With `mux_gain_mismatch` off ($\varepsilon_g=0$) this collapses to the matched-gain transport.

## Numerical method

N/A - the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| inter-leg gain mismatch | per-leg switch on-resistance / process offset between the two legs | static fractional gain mismatch $\varepsilon_g \sim \mathcal{N}(0,\sigma_{\varepsilon_g}^2)$, sampled once at fabricate; per-leg gain $g^{\pm}=g(1\pm\tfrac{1}{2}\varepsilon_g)$ — multiplicative, signal-dependent via $V_{\mathrm{cm}}$, state-independent (fabricate-time, sampled once per instance) | `mux_gain_mismatch_sigma_relative` | `mux_gain_mismatch` |
| common-mode noise | transport-path common-mode disturbance (thermal / $kT/C$ / coupling) | additive zero-mean Gaussian on both legs, state-independent | CM noise sigma | `mux_noise_cm` |
| differential-mode noise | transport-path differential disturbance (thermal / $kT/C$ / coupling) | additive zero-mean Gaussian, added to the positive leg and subtracted from the negative leg, state-independent | DM noise sigma | `mux_noise_dm` |

Inter-leg gain mismatch is **static** (sampled once per fabricate, per instance) and multiplicative; the two noise sources are **dynamic** (re-sampled every transport), zero-mean, and signal-independent. The mismatch sigma $\sigma_{\varepsilon_g}$ is a flat dimensionless config constant: the mux exposes no per-leg switch-device area, so it carries no Pelgrom area scaling. TODO (domain author): upgrade $\sigma_{\varepsilon_g}$ to the area-scaled Pelgrom form once a switch-device-area knob exists. For the dynamic terms, $kT/C$ sampling noise is the physical origin motivating the config-set CM/DM sigma in V; the mux config exposes no sampling capacitance $C$, so no derived variance $k_B T / C$ is asserted.

The dynamic terms are mathematically orthogonal to the static mismatch: a zero-mean additive sample has zero correlation with the input ($\mathbb{E}[n]=0$, $\mathbb{E}[n\,V_{\mathrm{cm}}]=0$), whereas the mismatch error $g\,\varepsilon_g\,V_{\mathrm{cm}}$ is a deterministic, input-correlated function of $V_{\mathrm{cm}}$; no additive sigma can represent it.

### Static inter-leg gain mismatch (CM-to-DM conversion)

A differential transport stage has two legs with insertion gains $g^{+}$ and $g^{-}$ that, with finite matching, differ. Write the matched (mean) gain and the fractional mismatch as

$$g = \tfrac{1}{2}(g^{+}+g^{-}), \qquad \varepsilon_g = \frac{g^{+}-g^{-}}{g},$$

and decompose the input into common and differential modes $V_{\mathrm{cm}} = \tfrac{1}{2}(V_{\mathrm{in}}^{+}+V_{\mathrm{in}}^{-})$, $V_{\mathrm{dm}} = V_{\mathrm{in}}^{+}-V_{\mathrm{in}}^{-}$. The transported differential value is then

$$V_{\mathrm{out}}^{+}-V_{\mathrm{out}}^{-} = g\,V_{\mathrm{dm}} + g\,\varepsilon_g\,V_{\mathrm{cm}},$$

so a leg-gain mismatch $\varepsilon_g$ leaks the common mode into the differential output with gain $g\,\varepsilon_g$. This is **common-mode-to-differential conversion**, the mechanism that sets a finite common-mode rejection ratio (CMRR). For this passive $R_{\mathrm{on}}$/load divider the converted differential error is $g\,\varepsilon_g\,V_{\mathrm{cm}}$ against a wanted differential gain $g$, so $\mathrm{CMRR} = 1/\varepsilon_g$ — the reciprocal of the fractional mismatch (the input-referred common-mode error $V_{\mathrm{cm}}/\mathrm{CMRR}$ is linear in $V_{\mathrm{cm}}$). Three properties make it incompatible with the additive-noise terms above:

1. **Static / deterministic** — fixed by fabrication for a given built device; it is the same on every transport, not re-sampled.
2. **Multiplicative and signal-dependent** — the error scales with $V_{\mathrm{cm}}$ (and, through any leg-dependent path, with the signal level); it vanishes only when $V_{\mathrm{cm}}=0$.
3. **Non-zero-mean over the signal ensemble** — because it tracks $V_{\mathrm{cm}}$, its mean over a non-zero-mean common-mode distribution is non-zero, unlike a zero-mean Gaussian.

Differential readout exists precisely to reject the common mode; the residual leakage $g\,\varepsilon_g\,V_{\mathrm{cm}}$ is the first-order error that survives that cancellation, which is why it is the static effect worth modelling first.

TODO (domain author): source $\sigma_{\varepsilon_g}$ from a process matching figure (e.g. a Pelgrom sigma for the switch pair) and set the CM/DM noise sigmas and any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `mux_gain` ($g$) | matched scalar transport gain | — | Design |
| `mux_gain_mismatch_sigma_relative` ($\sigma_{\varepsilon_g}$) | per-instance fractional inter-leg gain-mismatch sigma | — | Measured |
| CM noise sigma | common-mode transport-noise sigma | V | Measured |
| DM noise sigma | differential-mode transport-noise sigma | V | Measured |
| transport energy | per-transport dynamic energy | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the mux is a per-leg scalar-gain differential pass element with matched gain $g$ and a static fractional inter-leg gain mismatch $\varepsilon_g$ (CM-to-DM leakage); the only further departures modelled are the two additive zero-mean noise terms above.

Named omissions (out of scope, deferred for a stage operated in its linear, settled band): finite bandwidth / settling (single-pole $R_{\mathrm{on}}C$); signal-dependent $R_{\mathrm{on}}(V)$ path nonlinearity (harmonic distortion); the charge-injection pedestal $\Delta V = Q/C$ (signal-dependent, since the injected channel charge tracks the per-leg signal, so its residual after differential cancellation is signal-dependent, not a fixed offset) and clock feedthrough $\Delta V \approx V_{\mathrm{clk}}C_{gd}/(C_{gd}+C_L)$ (largely signal-independent, mostly common-mode with a static residual); off-isolation / channel-to-channel crosstalk (frequency-dependent capacitive coupling); and DC leakage ($V_{\mathrm{err}}=I_{\mathrm{LKG}}R_{\mathrm{load}}$, static additive offset). A static differential offset $V_{\mathrm{os}}$ is the natural companion of a future charge-injection model and is omitted until that signal-dependent source is modelled. The model is valid where the transported signal sits within the linear settled band and $\varepsilon_g$ is small enough that $g\,\varepsilon_g\,V_{\mathrm{cm}}$ stays below the readout resolution.

## Validation

TODO - link validation evidence once written.

## References

- P. R. Gray, P. J. Hurst, S. H. Lewis, R. G. Meyer, *Analysis and Design of Analog Integrated Circuits* — differential CMRR and CM-to-DM conversion from load/transconductance mismatch.
- B. Razavi, *Design of Analog CMOS Integrated Circuits* — finite CMRR from gain mismatch; signal-dependent on-resistance $R_{\mathrm{on}}(V)$ and the bootstrapped switch; charge injection / clock feedthrough.
- Analog Devices, MT-088, *Analog Switches and Multiplexers Basics* — $R_{\mathrm{on}}$/load attenuator gain error, $R_{\mathrm{on}}$ flatness/modulation, off-isolation, leakage, settling.
- Analog Devices, MT-042, *Op Amp Common-Mode Rejection Ratio (CMRR)* — finite CMRR from leg mismatch (e.g. 0.1% match → ~66 dB); input-referred common-mode error $V_{\mathrm{cm}}/\mathrm{CMRR}$, proportional to the common-mode signal.
- M. J. M. Pelgrom et al., *Matching properties of MOS transistors* — area-scaled device mismatch ($\sigma \propto 1/\sqrt{\mathrm{area}}$, mismatch shrinking with area).
- kT/C sampling noise (variance $k_B T / C$, zero-mean, white, signal-independent) — standard switched-capacitor noise analysis.

TODO (domain author): replace with the specific edition/section citations and a CIM-readout reference for the chosen $\varepsilon_g$ matching figure.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $g$ | matched scalar transport gain | — | `mux_gain` |
| $\varepsilon_g$ | fractional inter-leg gain mismatch (static, fabricate-fixed) | — | sampled at fabricate from `mux_gain_mismatch_sigma_relative` |
| $\sigma_{\varepsilon_g}$ | per-instance fractional gain-mismatch sigma | — | `mux_gain_mismatch_sigma_relative` |
| $V_{\mathrm{in}}^{\pm}, V_{\mathrm{out}}^{\pm}$ | input / output differential legs | V | transport input / output |
| $n_{\mathrm{cm}}$ | common-mode noise sample (zero-mean) | V | sampled in transport |
| $n_{\mathrm{dm}}$ | differential-mode noise sample (zero-mean) | V | sampled in transport |
| $V_{\mathrm{cm}}, V_{\mathrm{dm}}$ | common-/differential-mode of the input pair | V | derived from legs |

---

- **Internals**: [voltage_mux internals](../../internals/analog/voltage_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `VoltageMuxConfig` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
