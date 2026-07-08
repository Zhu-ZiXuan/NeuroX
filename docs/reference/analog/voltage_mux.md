# Voltage mux

The voltage mux transports a differential voltage pair, applying a matched per-leg transport gain with a static inter-leg gain mismatch and injecting common- and differential-mode transport noise, and it tallies per-transport energy and latency.

## Physical model

The model represents the mux as a per-leg scalar transport gain on a differential voltage pair plus two additive transport-noise terms, separating two physically distinct non-ideality families:

- **Static (fabrication-fixed, sampled once per instance).** The two legs' transport gains differ by a process offset. This **inter-leg gain mismatch** is modelled by promoting the matched gain $g$ to per-leg gains $g^{\pm} = g(1 \pm \tfrac{1}{2}\varepsilon_g)$, with $\varepsilon_g$ drawn once at fabrication; it is multiplicative and converts the input common mode into a signal-dependent differential error (finite CMRR, derived below).
- **Dynamic (re-sampled every transport).** Zero-mean, signal-independent thermal / $kT/C$ / coupling disturbances, modelled by the two additive noise terms below — the common-mode term shared by both legs, the differential-mode term antisymmetric. They do **not** model the static mismatch.

## Governing equations

For a differential input pair each leg is scaled by its own static insertion gain and perturbed by the two noise terms,

$$V_{\mathrm{out}}^{\pm} = g\,\Bigl(1 \pm \tfrac{1}{2}\varepsilon_g\Bigr)\,V_{\mathrm{in}}^{\pm} + n_{\mathrm{cm}} \pm n_{\mathrm{dm}},$$

where $g$ is the matched transport gain, $\varepsilon_g$ the fabrication-fixed fractional inter-leg gain mismatch, $n_{\mathrm{cm}}$ the common-mode noise sample (shared by both legs) and $n_{\mathrm{dm}}$ the differential-mode noise sample (added to the positive leg, subtracted from the negative leg); each noise term is zero-mean. Taking the leg difference,

$$V_{\mathrm{out}}^{+}-V_{\mathrm{out}}^{-} = g\,V_{\mathrm{dm}} + g\,\varepsilon_g\,V_{\mathrm{cm}} + 2\,n_{\mathrm{dm}},$$

with $V_{\mathrm{cm}} = \tfrac{1}{2}(V_{\mathrm{in}}^{+}+V_{\mathrm{in}}^{-})$ and $V_{\mathrm{dm}} = V_{\mathrm{in}}^{+}-V_{\mathrm{in}}^{-}$: the wanted differential gain $g\,V_{\mathrm{dm}}$, the static common-mode-to-differential leakage $g\,\varepsilon_g\,V_{\mathrm{cm}}$, and the dynamic differential noise. With $\varepsilon_g=0$ this collapses to the matched-gain transport.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| inter-leg gain mismatch | process offset between the two legs | static fractional gain mismatch $\varepsilon_g \sim \mathcal{N}(0,\sigma_{\varepsilon_g}^2)$, sampled once at fabrication; per-leg gain $g^{\pm}=g(1\pm\tfrac{1}{2}\varepsilon_g)$ — multiplicative, signal-dependent via $V_{\mathrm{cm}}$, sampled once per instance | `mux_gain_mismatch_sigma_relative` |
| common-mode noise | transport-path common-mode disturbance (thermal / $kT/C$ / coupling) | additive zero-mean Gaussian on both legs, signal-independent | `mux_noise_cm_sigma__V` |
| differential-mode noise | transport-path differential disturbance (thermal / $kT/C$ / coupling) | additive zero-mean Gaussian, added to the positive leg and subtracted from the negative leg, signal-independent | `mux_noise_dm_sigma__V` |

Inter-leg gain mismatch is **static** (sampled once at fabrication, per instance) and multiplicative; the two noise sources are **dynamic** (re-sampled every transport), zero-mean, and signal-independent. The mismatch sigma $\sigma_{\varepsilon_g}$ is a flat dimensionless constant: the model carries no per-leg device-area parameter, so it applies no Pelgrom area scaling. TODO (domain author): upgrade $\sigma_{\varepsilon_g}$ to the area-scaled Pelgrom form once a switch-device-area knob exists. For the dynamic terms, $kT/C$ sampling noise is the physical origin motivating the CM/DM sigma in V; the model carries no sampling capacitance $C$, so no derived variance $k_B T / C$ is asserted. The shared area-scaled mismatch and $kT/C$ sampling laws are in [nonideality](../nonideality.md).

The dynamic terms are mathematically orthogonal to the static mismatch: a zero-mean additive sample has zero correlation with the input ($\mathbb{E}[n]=0$, $\mathbb{E}[n\,V_{\mathrm{cm}}]=0$), whereas the mismatch error $g\,\varepsilon_g\,V_{\mathrm{cm}}$ is a deterministic, input-correlated function of $V_{\mathrm{cm}}$; no additive sigma can represent it.

### Static inter-leg gain mismatch (CM-to-DM conversion)

Writing the matched (mean) gain and the fractional mismatch as

$$g = \tfrac{1}{2}(g^{+}+g^{-}), \qquad \varepsilon_g = \frac{g^{+}-g^{-}}{g},$$

the transported differential value $V_{\mathrm{out}}^{+}-V_{\mathrm{out}}^{-} = g\,V_{\mathrm{dm}} + g\,\varepsilon_g\,V_{\mathrm{cm}}$ leaks the common mode into the differential output with gain $g\,\varepsilon_g$. This **common-mode-to-differential conversion** sets a finite common-mode rejection ratio $\mathrm{CMRR} = 1/\varepsilon_g$, the reciprocal of the fractional mismatch, so the input-referred common-mode error $V_{\mathrm{cm}}/\mathrm{CMRR}$ is linear in $V_{\mathrm{cm}}$. This residual leakage $g\,\varepsilon_g\,V_{\mathrm{cm}}$ is the first-order differential error that survives common-mode cancellation, so it is the static effect the model captures first.

TODO (domain author): source $\sigma_{\varepsilon_g}$ from a process matching figure (e.g. a Pelgrom sigma for the switch pair) and set the CM/DM noise sigmas and any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `mux_gain` ($g$) | matched scalar transport gain | — | $> 0$ | Design |
| `mux_gain_mismatch_sigma_relative` ($\sigma_{\varepsilon_g}$) | per-instance fractional inter-leg gain-mismatch sigma | — | $\geq 0$ | Measured |
| `mux_noise_cm_sigma__V` | common-mode transport-noise sigma | V | $\geq 0$ | Measured |
| `mux_noise_dm_sigma__V` | differential-mode transport-noise sigma | V | $\geq 0$ | Measured |
| `energy_per_access__fJ` | per-transport dynamic energy | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | per-transport latency | ns | $\geq 0$ | Design |
| `area_per_inst__um2`, `leakage_per_inst__uW` | static PPA fields | um^2, uW | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $g$ | matched scalar transport gain | — | `mux_gain` |
| $\varepsilon_g$ | fractional inter-leg gain mismatch (static, fabrication-fixed) | — | `eps_g` |
| $\sigma_{\varepsilon_g}$ | per-instance fractional gain-mismatch sigma | — | `mux_gain_mismatch_sigma_relative` |
| $V_{\mathrm{in}}^{\pm}$ | input differential legs | V | `v_pos__V`, `v_neg__V` |
| $V_{\mathrm{out}}^{\pm}$ | output differential legs | V | `v_pos_muxed__V`, `v_neg_muxed__V` |
| $n_{\mathrm{cm}}$ | common-mode noise sample (zero-mean) | V | `n_cm__V` |
| $n_{\mathrm{dm}}$ | differential-mode noise sample (zero-mean) | V | `n_dm__V` |
| $V_{\mathrm{cm}}, V_{\mathrm{dm}}$ | common-/differential-mode of the input pair | V | — |

## Assumptions, scope & validity

Stated assumption: the mux is a per-leg scalar-gain differential pass element with matched gain $g$ and a static fractional inter-leg gain mismatch $\varepsilon_g$ (CM-to-DM leakage); the only further departures modelled are the two additive zero-mean noise terms above.

Named omissions (out of scope, deferred for a stage operated in its linear, settled band): finite bandwidth / settling (single-pole $R_{\mathrm{on}}C$); signal-dependent $R_{\mathrm{on}}(V)$ path nonlinearity (harmonic distortion); the charge-injection pedestal $\Delta V = Q/C$ (signal-dependent, since the injected channel charge tracks the per-leg signal, so its residual after differential cancellation is signal-dependent, not a fixed offset) and clock feedthrough $\Delta V \approx V_{\mathrm{clk}}C_{gd}/(C_{gd}+C_L)$ (largely signal-independent, mostly common-mode with a static residual); off-isolation / channel-to-channel crosstalk (frequency-dependent capacitive coupling); and DC leakage ($V_{\mathrm{err}}=I_{\mathrm{LKG}}R_{\mathrm{load}}$, static additive offset). A static differential offset $V_{\mathrm{os}}$ is omitted; it would pair with a charge-injection model, which is not currently modelled. The model is valid where the transported signal sits within the linear settled band and $\varepsilon_g$ is small enough that $g\,\varepsilon_g\,V_{\mathrm{cm}}$ stays below the readout resolution.

## Validation

TODO — link validation evidence once written.

## References

- P. R. Gray, P. J. Hurst, S. H. Lewis, R. G. Meyer, *Analysis and Design of Analog Integrated Circuits* — differential CMRR and CM-to-DM conversion from load/transconductance mismatch.
- B. Razavi, *Design of Analog CMOS Integrated Circuits* — finite CMRR from gain mismatch; signal-dependent on-resistance $R_{\mathrm{on}}(V)$ and the bootstrapped switch; charge injection / clock feedthrough.
- Analog Devices, MT-088, *Analog Switches and Multiplexers Basics* — $R_{\mathrm{on}}$/load attenuator gain error, $R_{\mathrm{on}}$ flatness/modulation, off-isolation, leakage, settling.
- Analog Devices, MT-042, *Op Amp Common-Mode Rejection Ratio (CMRR)* — finite CMRR from leg mismatch (e.g. 0.1% match → ~66 dB); input-referred common-mode error $V_{\mathrm{cm}}/\mathrm{CMRR}$, proportional to the common-mode signal.
- M. J. M. Pelgrom et al., *Matching properties of MOS transistors* — area-scaled device mismatch ($\sigma \propto 1/\sqrt{\mathrm{area}}$, mismatch shrinking with area).
- kT/C sampling noise (variance $k_B T / C$, zero-mean, white, signal-independent) — standard switched-capacitor noise analysis.

TODO (domain author): replace with the specific edition/section citations and a CIM-readout reference for the chosen $\varepsilon_g$ matching figure.

---

- **Internals**: [voltage_mux internals](../../internals/analog/voltage_mux.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VoltageMuxConfig` (see `api`)
