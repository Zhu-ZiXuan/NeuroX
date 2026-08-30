# Non-idealities

A non-ideality replaces a nominal quantity with a sample drawn from a distribution centred on (additive) or scaled around (multiplicative) it. This document collects the shared statistical laws those samples obey; each subsystem parameterises its own sources against these laws.

## Scope

Only the general, cross-device statistical laws live here: the area-scaled mismatch of fabricated devices and the thermal noise of charge sampling. Device-specific non-idealities are parameterised in that device's own reference and are not restated here.

## Physical origins

Every perturbation belongs to one of two classes, distinguished by fixedness — whether its sample is frozen or resampled:

- **Static** — fixed once and held until the device is re-fabricated or re-programmed. It covers fabrication mismatch, from spatially-uncorrelated microscopic process fluctuations averaged over the device area, together with the programming write deviation left by a finite-precision write.
- **Dynamic** — resampled at every access, set by the operating point and temperature at that access rather than frozen. It covers thermal, flicker, random-telegraph, and charge-sampling ($kT/C$) fluctuations.

A source can be static yet device-specific, like programming variation.

## State dependence of the spread

Orthogonal to that static or dynamic origin, the spread $\sigma$ of a source is obtained in one of two ways; each subsystem classifies its own sources accordingly:

- **State-independent** — $\sigma$ is a fixed constant and the same distribution is sampled at every element. Used where the magnitude does not track the signal (e.g. comparator thermal noise, stuck-at faults).
- **State-dependent** — $\sigma$ is derived per element from the operating state, either the signal itself or an auxiliary state variable. Used where the magnitude tracks the conductance state (e.g. programming variability) or scales with the device area (e.g. Pelgrom mismatch, $kT/C$ sampling, both shrinking as $1/\sqrt{\mathrm{area}}$).

## Additive and multiplicative perturbation

Independently of both classifications, a draw either adds to the nominal quantity or scales it,

$$x = x^{\mathrm{nom}} + \varepsilon, \qquad x = x^{\mathrm{nom}} (1 + \eta),$$

with $\varepsilon$ and $\eta$ zero-mean, so both forms leave the mean at $x^{\mathrm{nom}}$. The additive form suits a quantity whose spread is set by the surrounding circuit rather than by the value itself — an input-referred comparator offset, a $kT/C$ sampling fluctuation — and its $\sigma$ carries the unit of $x$. The multiplicative (relative) form suits a quantity whose spread tracks its own magnitude, such as a reference tap or a mirror ratio: one dimensionless $\sigma$ then covers values of differing magnitude, where the additive form would need one $\sigma$ per value. The multiplicative form also has an exact fixed point at zero — a nominal $0$ stays exactly $0$ under any relative $\sigma$ — which is what keeps a ground or rail reference tap stable and exact under noise.

A disabled source is the exact identity, not a draw with $\sigma = 0$: the nominal quantity passes through unperturbed and unchanged, so switching every source off reproduces the ideal model exactly.

## Pelgrom area-scaled mismatch

The two-term Pelgrom law for the spread of a parameter difference $\Delta P$ between two matched devices is

$$\sigma^2(\Delta P) = \frac{A_P^2}{W L} + S_P^2\, D^2,$$

a local area term ($A_P$ over the gate-region product $W L$) plus a separate long-range gradient term ($S_P$ scaling the device separation $D$). The local term vanishes monotonically as the device grows — there is no floor. The only non-vanishing large-area residual is the area-independent gradient term $S_P^2 D^2$, an additive variance contribution rather than a floor on $\sigma$.

### Absolute and relative framing

The same $1/\mathrm{area}$ variance is written in whichever framing matches the parameter. An intensive parameter that adds directly — a threshold voltage — uses the absolute spread, $\sigma(\Delta V_{\mathrm{th}}) \propto 1/\sqrt{W L}$. A multiplicative parameter — a current factor, or a capacitance that perturbs its operand proportionally — uses the relative spread. For a capacitor, with $C_k$ the element capacitance and $C_{\mathrm{unit}}$ the unit cell, the relative form is

$$\sigma\!\left(\frac{\Delta C_k}{C_k}\right) = \sigma_{\mathrm{rel}} \sqrt{\frac{C_{\mathrm{unit}}}{C_k}},$$

so the relative spread is $\propto 1/\sqrt{C_k}$ and shrinks with area, while the equivalent absolute spread $\sigma(\Delta C_k) = \sigma_{\mathrm{rel}}\sqrt{C_{\mathrm{unit}}\, C_k}$ grows as $\sqrt{C_k}$. The two framings of one variance look opposite only because one is normalised by $C_k$ and the other is not.

## Thermal charge-sampling noise

Sampling a voltage onto a capacitor $C$ at temperature $T$ leaves a thermal ($kT/C$) uncertainty of voltage variance

$$\sigma^2 = \frac{k_B T}{C},$$

a state-dependent source whose spread shrinks as $1/\sqrt{C}$, i.e. with area, like the mismatch above.

## Parameter ownership

No parameters belong to the shared laws themselves. Each source's distribution parameters — its $\sigma$, the Pelgrom coefficients $A_P$ and $S_P$, the relative spread $\sigma_{\mathrm{rel}}$, and the unit-cell capacitance $C_{\mathrm{unit}}$ — belong to the owning subsystem's §Parameters table, with provenance per [module_parameter](../../conventions/module_parameter.md). The Boltzmann constant $k_B$ and the temperature $T$ are shared across subsystems and defined in [notation_conventions](../../conventions/notation_conventions.md).

TODO (domain author): the concrete parameterisation of the long-range gradient term — the $S_P$ coefficient, the $D$ separation, and the subsystem that owns them.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $\sigma$ | distribution spread (standard deviation) of a source | varies | — |
| $\varepsilon$ | additive perturbation of a nominal quantity | same as the quantity | — |
| $\eta$ | relative (multiplicative) perturbation of a nominal quantity | — | — |
| $\Delta P$ | parameter difference between two matched devices | varies | — |
| $A_P$ | Pelgrom local-area mismatch coefficient | param × um | — |
| $W, L$ | device gate width and length | um | — |
| $S_P$ | Pelgrom long-range gradient coefficient | param / um | — |
| $D$ | separation between two matched devices | um | — |
| $V_{\mathrm{th}}$ | threshold voltage (intensive-mismatch example) | V | — |
| $C_k$ | element capacitance | fF | — |
| $C_{\mathrm{unit}}$ | unit-cell capacitance | fF | — |
| $\sigma_{\mathrm{rel}}$ | relative mismatch spread at the unit cell | — | `sigma_relative` |
| $k_B$ | Boltzmann constant | J/K | `K_BOLTZMANN__J_per_K` |
| $T$ | absolute temperature (runtime input) | K | `T__K` |
| $C$ | sampling capacitance | fF | — |

## Assumptions and validity

- Static mismatch is a per-instance constant fixed at fabrication; dynamic noise is resampled per read and is i.i.d. across reads.
- Microscopic fluctuations are spatially uncorrelated and area-averaged, so the local Pelgrom term has variance $\propto 1/\mathrm{area}$.
- Spatial correlation between elements is modelled only through the long-range gradient term; every other draw is independent per element.

TODO (domain author): the device-size and temperature ranges over which the Pelgrom and $kT/C$ laws hold, the regimes where inter-element spatial correlation must be modelled explicitly, and confirmation of the area-scaling direction of the absolute-vs-relative framing.
