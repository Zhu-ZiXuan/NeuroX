# Non-idealities

A non-ideality perturbs a nominal quantity under a statistical law. Shared laws include area-scaled fabrication mismatch and thermal charge-sampling noise; each subsystem specifies its sources' parameters.

## Physical origins

Perturbations have two lifetimes:

- **Static** — held until fabrication or programming is repeated, including fabrication mismatch and programming write deviation.
- **Dynamic** — resampled per access at its operating point and temperature, including thermal, flicker, random-telegraph, and charge-sampling ($kT/C$) fluctuations.

## State dependence of the spread

Independently of lifetime, a source's spread $\sigma$ is:

- **State-independent** — constant across elements and operating states.
- **State-dependent** — derived per element from its signal or another state variable, such as conductance or device area.

## Additive and multiplicative perturbation

A perturbation adds to or scales the nominal quantity:

$$x = x^{\mathrm{nom}} + \varepsilon, \qquad x = x^{\mathrm{nom}} (1 + \eta),$$

Zero-mean $\varepsilon$ and $\eta$ preserve the mean $x^{\mathrm{nom}}$. Additive spread carries the unit of $x$; relative spread is dimensionless and scales the absolute variation with the nominal magnitude. Multiplicative perturbation preserves a nominal zero exactly.

A disabled source preserves the nominal quantity exactly. Disabling all sources recovers the ideal model.

## Pelgrom area-scaled mismatch

The two-term Pelgrom law for the spread of a parameter difference $\Delta P$ between two matched devices is

$$\sigma^2(\Delta P) = \frac{A_P^2}{W L} + S_P^2\, D^2,$$

The local term vanishes as gate area $W L$ grows. The long-range gradient term depends on device separation $D$ and contributes the area-independent variance $S_P^2D^2$.

### Absolute and relative framing

An additive intensive parameter uses absolute spread, as in $\sigma(\Delta V_{\mathrm{th}}) \propto 1/\sqrt{W L}$. Multiplicative parameters use relative spread. For element capacitance $C_k$ and unit-cell capacitance $C_{\mathrm{unit}}$,

$$\sigma\!\left(\frac{\Delta C_k}{C_k}\right) = \sigma_{\mathrm{rel}} \sqrt{\frac{C_{\mathrm{unit}}}{C_k}},$$

The relative spread shrinks as $1/\sqrt{C_k}$, while the equivalent absolute spread $\sigma(\Delta C_k) = \sigma_{\mathrm{rel}}\sqrt{C_{\mathrm{unit}}\, C_k}$ grows as $\sqrt{C_k}$.

## Thermal charge-sampling noise

Sampling a voltage onto a capacitor $C$ at temperature $T$ leaves a thermal ($kT/C$) uncertainty of voltage variance

$$\sigma^2 = \frac{k_B T}{C},$$

a state-dependent source whose spread shrinks as $1/\sqrt{C}$.

## Parameter ownership

Source parameters such as $\sigma$, $A_P$, $S_P$, $\sigma_{\mathrm{rel}}$, and $C_{\mathrm{unit}}$ belong to their subsystems, with provenance per [module_parameter](../../conventions/module_parameter.md). The Boltzmann constant is defined under [physical constants](physics.md#physical-constants); temperature is a runtime input.

The implemented mismatch kernels use the local area-scaled term. Distance-dependent gradient noise is not implemented.

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
| $k_B$ | Boltzmann constant | fJ/K | `K_BOLTZMANN__fJ_per_K` |
| $T$ | absolute temperature (runtime input) | K | `T__K` |
| $C$ | sampling capacitance | fF | — |

## Assumptions and validity

- Static mismatch is a per-instance constant fixed at fabrication; dynamic noise is resampled per read and is i.i.d. across reads.
- Microscopic fluctuations are spatially uncorrelated and area-averaged, so the local Pelgrom term has variance $\propto 1/\mathrm{area}$.
- Kernel draws are independent per element. Correlation from a shared physical source is represented by reusing its sampled state across consumers.

The applicable device-size and temperature ranges and any additional spatial correlation require model-specific characterization.
