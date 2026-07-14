# MCS SAR ADC

A $V_{\mathrm{cm}}$-based (Merged Capacitor Switching, MCS) differential SAR ADC. Each conversion digitizes the differential legs $V^{+}, V^{-}$ against a mode-selected reference $V_{\mathrm{ref}}$ at a runtime resolution $b \leq b_{\max}$, returning a signed integer code; the sign (MSB) is resolved by a free top-plate comparison and each remaining bit by a merged-capacitor charge-redistribution step.

## Physical model

A differential MCS topology with bottom-plate sampling and two independent binary-weighted CDAC arrays, one per input leg. Physical CDAC depth is $b_{\max}$ bits; the active array has $b_{\max}-1$ binary-weighted caps plus a dummy unit cap. The scheme uses three bottom-plate levels — $V_{\mathrm{ref}}$, the common-mode "third reference" $V_{\mathrm{cm}} = V_{\mathrm{ref}}/2$, and ground. Each single-ended leg swings rail-to-rail, $V^{+}, V^{-} \in [0,\ V_{\mathrm{ref}}]$, so the differential input ranges over $V^{+} - V^{-} \in [-V_{\mathrm{ref}},\ +V_{\mathrm{ref}}]$. The dummy cap is a unit cap equal to the LSB cap (tied-smallest), making the array total a power of two, $C_{\mathrm{total}} = 2^{\,b_{\max}-1}\,C_{\mathrm{unit}}$; it samples and charge-divides like every other cap.

A conversion proceeds in three phases:

1. **Sample** - the bottom plates of the two CDAC arrays track the positive and negative inputs while the top plates float; on release the bottom plates snap to the common-mode voltage $V_{\mathrm{cm}}$. By charge conservation the floating top plate holds the primitive value $V_{\mathrm{top}} = 2V_{\mathrm{cm}} - V_{\mathrm{in}}$, which specialises to $V_{\mathrm{ref}} - V_{\mathrm{in}}$ at the design point $V_{\mathrm{cm}} = V_{\mathrm{ref}}/2$.
2. **MSB decision** - a free comparison of the two top plates resolves the most-significant bit; no MSB cap is needed because the differential top plates hold $V_{\mathrm{top}}^{+} - V_{\mathrm{top}}^{-} = -(V^{+} - V^{-})$, independent of $V_{\mathrm{cm}}$, so the sign is resolved with no capacitor switched. The comparison carries the static comparator offset and per-cycle thermal noise.
3. **SAR loop** - each subsequent cycle switches one cap on each leg from $V_{\mathrm{cm}}$ to $V_{\mathrm{ref}}$ or to ground according to the previous bit, moving each top plate by the primitive step $\pm (V_{\mathrm{ref}} - V_{\mathrm{cm}})\,C_k/C_{\mathrm{total}}$, equal to $\pm V_{\mathrm{cm}}\,C_k/C_{\mathrm{total}}$ at the design point, until $b$ bits are resolved. The toggled bottom plate swings only $|V_{\mathrm{ref}} - V_{\mathrm{cm}}| = V_{\mathrm{cm}} = V_{\mathrm{ref}}/2$ (half $V_{\mathrm{ref}}$) — the origin of the MCS energy saving.

The modelled non-idealities are static per-cap Pelgrom mismatch (independent positive / negative legs), a static comparator offset, per-SAR-cycle comparator thermal noise, and kT/C sampling noise on the held top plates.

## Governing equations

Each SAR cycle perturbs the differential top-plate voltage by the merged-capacitor step, in primitive form

$$\Delta V_{\mathrm{top},k} = \pm\, (V_{\mathrm{ref}} - V_{\mathrm{cm}})\,\frac{C_k}{C_{\mathrm{total}}} \;\xrightarrow{V_{\mathrm{cm}} = V_{\mathrm{ref}}/2}\; \pm\, V_{\mathrm{cm}}\,\frac{C_k}{C_{\mathrm{total}}} = \pm\, \frac{V_{\mathrm{ref}}}{2}\,\frac{C_k}{C_{\mathrm{total}}},$$

where $C_k$ is the capacitance of the cap switched on cycle $k$, carrying the static per-cap Pelgrom mismatch (independent legs), and $C_{\mathrm{total}}$ the array total. The $V_{\mathrm{cm}}$ form is an identity valid only because $V_{\mathrm{ref}} - V_{\mathrm{cm}} = V_{\mathrm{cm}}$ at the design point; the bottom-plate swing is $V_{\mathrm{ref}}/2$ (half $V_{\mathrm{ref}}$). The MSB is the sign of the free differential comparison; each subsequent bit is the sign of the running differential after the cycle's step, accumulated into the unsigned code. The unsigned code is then clamped to $[0,\ 2^{b}-1]$ and shifted by the zero code $2^{\,b-1}$ to the signed range $[-2^{\,b-1},\ 2^{\,b-1}-1]$, following the [family](family.md#signed-code-range) signed-code convention. The zero code $2^{\,b-1}$ is the bucket midpoint of the symmetric differential design, where $V^{+} - V^{-} = 0$ sits centred between the rail-symmetric extremes $\pm V_{\mathrm{ref}}$, justifying the symmetric zero point.

## Numerical method

The conversion performs $b$ comparisons total — one free MSB comparison (no cap switched) plus $b-1$ in-loop decision cycles, each a single cap switch per leg followed by a comparison, giving $b-1$ switch events in all. Each in-loop cycle is a closed-form step and a sign decision, with no inner iteration.

## Energy model

Per-conversion energy is the sum of:

- one-shot sampling energy: the bottom plates track $V_{\mathrm{in}}$ during sample and snap to $V_{\mathrm{cm}}$ on release;
- per-cycle MCS switching energy (derived below);
- reset energy dissipating the residual differential charge left on the two arrays (derived below);
- a lump-sum overhead $E_{\mathrm{bootstrap}} + b\cdot E_{\mathrm{const}/\mathrm{bit}}$.

### Switching energy

The reference-drawn switching energy of cycle $k$ is the rail voltage times the signed charge the $V_{\mathrm{ref}}$ rail sources, $E_k = V_{\mathrm{ref}}\,\Delta Q_{\mathrm{ref},k}$. The derivation follows the operation, the resulting state change, and the energy in turn:

1. **Sample** — bottom plate at $V_{\mathrm{in}}$, top plate biased to $V_{\mathrm{cm}}$ then released; the floating top plate stores $Q = C_{\mathrm{total}}\,(V_{\mathrm{cm}} - V_{\mathrm{in}})$.
2. **Free MSB** — bottom plates snap from $V_{\mathrm{in}}$ to $V_{\mathrm{cm}}$; charge conservation gives $V_{\mathrm{top}} = 2V_{\mathrm{cm}} - V_{\mathrm{in}}$, and the differential pair $-(V^{+} - V^{-})$ is $V_{\mathrm{cm}}$-independent. No cap is switched and no reference charge flows, so $E_1 = 0$.
3. **In-loop cycle $k$** — exactly one cap of weight $C_k$ toggles per leg: on the leg whose top plate is higher its bottom plate moves $V_{\mathrm{cm}} \to \mathrm{GND}$, the mirror cap moves $V_{\mathrm{cm}} \to V_{\mathrm{ref}}$, pinning the DAC common mode at $V_{\mathrm{cm}}$. With top-plate voltages $V_{\mathrm{top},k}^{-}, V_{\mathrm{top},k}^{+}$ just before/after the toggle (related by the step $\Delta V_{\mathrm{top},k}$ above), the rail-sourced charge is the signed

$$\Delta Q_{\mathrm{ref},k} = C_k\big[(V_{\mathrm{ref}} - V_{\mathrm{top},k}^{+}) - (V_{\mathrm{cm}} - V_{\mathrm{top},k}^{-})\big].$$

It is negative on charge-recycling transitions, where charge is pushed back to the rail, so $E_k < 0$ there.

The signed per-step energy follows. Writing the comparison stage as $i$ (with $i = 1$ the free MSB, $E_1 = 0$, decided bits $D_j \in \{0,1\}$, and $N = b$),

$$E_i = \left[\,2^{\,N-i-2} \;-\; 2\,D_{i-1}\!\!\sum_{j=1}^{i-1}(2D_j - 1)\,2^{\,N-i-j-2}\,\right]\,C_{\mathrm{unit}}\,V_{\mathrm{ref}}^2, \qquad i = 2,\dots,b.$$

The leading $2^{\,N-i-2}$ is the half-swing self-charging of the freshly toggled cap; the bracketed sum over prior bits $D_j$ is the signed cross-coupling from caps already set, with $(2D_j - 1) = \pm 1$ flipping sign per prior bit, so the bracket — hence $E_i$ — is negative on recycling steps. Equivalently, in the schematic-with-explicit-factor form

$$E_k = \tfrac{1}{2}\,V_{\mathrm{ref}}^2\,\frac{C_k}{C_{\mathrm{total}}}\,f_k(b_1\dots b_{k-1}),$$

the half-swing factor $(V_{\mathrm{cm}}/V_{\mathrm{ref}})^2 = 1/4$ is already absorbed (the toggled bottom plate swings only $V_{\mathrm{ref}}/2$), and $f_k$ is the signed $\pm 1$-weighted combination of previously decided bits — equal to the bracket divided by $2^{\,N-i-2}$. The conventional non-negative form $\tfrac{1}{2}V_{\mathrm{ref}}^2 C_k(1 - C_k/C_{\mathrm{total}})$ uses the full $V_{\mathrm{ref}}$ excursion and cannot represent these negative recycling steps.

Because the per-step $E_i$ depends on the specific output code, the citable spec quantity is the equiprobable-code average, with the unit-cap normalisation $C_{\mathrm{total}} = 2^{\,b-1}\,C_{\mathrm{unit}}$ per single-ended binary array fixed as a prerequisite:

$$\overline{E}_{\mathrm{sw}} = \left(\sum_{i=1}^{b-1} 2^{\,b-2-2i}\,(2^{\,i} - 1)\right) C_{\mathrm{unit}}\,V_{\mathrm{ref}}^2.$$

The upper limit $b-1$ reflects the free MSB plus the $b-1$ switched bits. At $b = 10$ this evaluates to $\overline{E}_{\mathrm{sw}} = 170.2\,C_{\mathrm{unit}}V_{\mathrm{ref}}^2$, 87.52% below the conventional baseline $1363.3\,C_{\mathrm{unit}}V_{\mathrm{ref}}^2$ (this is the generic tri-level $V_{\mathrm{cm}}$-based number, not a monotonic-variant figure). Both numbers are reproducible only within this one unit-cap normalisation. The code average is a topology-level comparative figure, not a precise per-conversion power: it is sensitive to the unit-cap normalisation, the $(V_{\mathrm{ref}} - V_{\mathrm{cm}})$ half-swing assumption, and ideal settling.

### Reset energy

After the SAR loop the two CDAC arrays hold a residual differential charge, dissipated when they reset to the sampling configuration for the next conversion. Summing the signed leg-to-leg capacitance difference over the $b-1$ switched caps,

$$C_{\mathrm{diff}} = \sum_{k=1}^{b-1}\left(\tfrac{1}{2} - D_k\right)\left(C_{p,k} - C_{n,k}\right),$$

with $D_k \in \{0,1\}$ the decided bit and $C_{p,k}, C_{n,k}$ the positive- and negative-leg instances of the switched cap $C_k$ (independent Pelgrom mismatch), the reset dissipation is

$$E_{\mathrm{reset}} = \tfrac{1}{2}\,V_{\mathrm{ref}}^2\,|C_{\mathrm{diff}}|.$$

It vanishes for perfectly matched legs ($C_{p,k} = C_{n,k}$): the term is nonzero only through the differential cap mismatch.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| cap mismatch | per-cap area/oxide variation | static per-cap Pelgrom Gaussian on $C_k$ (independent legs), at fabricate | `cap_mismatch_sigma_relative` |
| comparator offset | static comparator input offset | static Gaussian threshold offset, at fabricate | `comparator_offset_sigma__V` |
| comparator thermal noise | per-decision thermal noise | additive Gaussian per SAR cycle | `comparator_thermal_noise_sigma__V` |
| sampling thermal noise | kT/C noise on held top plates | additive Gaussian at sample, sigma set by $T$ | (derived from $T$, $C$) |
| quantization | intrinsic SAR resolution | deterministic | $b$ |

The comparator thermal-noise sigma scales as $\sqrt{T}$ anchored at 300 K.

The kT/C sampling noise (the additive-Gaussian row above) imprints on each held top plate a perturbation of variance

$$\sigma_V^2 = \frac{k_B T}{C_{\mathrm{total}}},$$

with $k_B$ the Boltzmann constant, $T$ the operating temperature, and $C_{\mathrm{total}}$ the sampling leg's array total. The two legs are perturbed independently, each with its own $C_{\mathrm{total}}$.

TODO (domain author): citations for the MCS switching-energy and Pelgrom models.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `max_bits` | physical CDAC depth $b_{\max}$ | — | $\geq 2$ | Design |
| `clk_period__ns` | SAR clock period | ns | $> 0$ | Design |
| `c_unit__fF` | unit-cap capacitance | fF | $> 0$ | Design |
| `cap_mismatch_sigma_relative` | per-cap Pelgrom mismatch sigma | — | $\geq 0$ | Measured |
| `comparator_offset_sigma__V` | static comparator-offset sigma | V | $\geq 0$ | Measured |
| `comparator_thermal_noise_sigma__V` | per-cycle comparator-noise sigma (at 300 K) | V | $\geq 0$ | Measured |
| `e_bootstrap__fJ` | per-conversion bootstrap energy | fJ | $\geq 0$ | Design |
| `e_constant_per_bit__fJ` | per-bit constant energy overhead | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

The reference voltage is not a parameter of this ADC — it is one of the reference taps supplied per conversion (see [family](family.md)). A conversion spans $b+1$ clock periods (one sample cycle plus $b$ comparison cycles), so its latency is $(b+1)\cdot$ `clk_period__ns`. Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V^{+}, V^{-}$ | differential input legs | V | `v_pos__V`, `v_neg__V` |
| $V_{\mathrm{cm}}$ | common-mode third reference, $V_{\mathrm{ref}}/2$ | V | derived |
| $V_{\mathrm{ref}}$ | reference voltage | V | `v_refs__V[..., mode]` |
| $V_{\mathrm{in}}$ | sampled input on a leg | V | sampled in `convert` |
| $C_k$ | capacitance of cap $k$ (mismatched after fabricate) | fF | `c_p__fF`, `c_n__fF` |
| $C_{\mathrm{total}}$ | total array capacitance, $2^{\,b_{\max}-1}C_{\mathrm{unit}}$ | fF | derived |
| $C_{\mathrm{unit}}$ | unit-cap capacitance | fF | `c_unit__fF` |
| $C_{\mathrm{diff}}$ | residual differential capacitance after the SAR loop | fF | `c_diff__fF` |
| $b$ | resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | physical CDAC depth | — | `max_bits` |
| $E_k$ | signed per-cycle MCS switching energy | fJ | energy accounting |
| $\overline{E}_{\mathrm{sw}}$ | equiprobable-code-average switching energy | fJ | energy accounting |
| $f_k$ | signed prior-bit factor in $E_k$ | — | energy accounting |
| $E_{\mathrm{reset}}$ | reset-phase residual-charge dissipation | fJ | energy accounting |
| $E_{\mathrm{bootstrap}}, E_{\mathrm{const}/\mathrm{bit}}$ | energy overheads | fJ | `e_bootstrap__fJ`, `e_constant_per_bit__fJ` |
| $T$ | operating temperature | K | `T__K` |
| $k_B$ | Boltzmann constant | J/K | `K_BOLTZMANN__J_per_K` |
| $\sigma_V$ | kT/C sampling-noise sigma on a held top plate | V | derived from $T$, $C_{\mathrm{total}}$ |

## Assumptions, scope & validity

Stated assumptions:

- The differential topology resolves the MSB by free comparison, so no dedicated MSB cap is modelled.
- For $b < b_{\max}$ the unreached smaller caps still sample and charge-divide — contributing to $C_{\mathrm{total}}$ and hence the step denominator — but are not switched, so they add no switching energy.

TODO (domain author): the validity range of the merged-capacitor step model (settling, parasitic coupling) and the operating envelope over which calibration is trusted.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the merged-capacitor-switching SAR topology and its energy model.

---

- **Internals**: [mcs_sar internals](../../../../internals/primitive/analog/adc/mcs_sar.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `McsSarAdcConfig`, `McsSarAdcPolicy` (see `api`)
