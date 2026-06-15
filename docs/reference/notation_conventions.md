# Notation & Conventions

The shared vocabulary for reading every equation in the [Reference](README.md) manual. Document format (unicode whitelist, LaTeX, code blocks) is in [contributing/doc_style](../contributing/doc_style.md); code-side conventions (variable naming, unit-encoding suffixes, dtype) are in [contributing/code_style](../contributing/code_style.md) and [contributing/naming_conventions](../contributing/naming_conventions.md).

Each document lists every symbol it uses in its own Symbols table (with the code field). This page pins the symbols **reused across subsystems** so one physical quantity keeps one symbol everywhere. Introduce a symbol only when it appears in an equation; a quantity that only labels a structure or count is written by its code field, not a symbol.

## Electrical and physical quantities

| Quantity | Symbol | Unit | Notes |
|---|---|---|---|
| voltage | $V$ | V | subscript names the node, e.g. $V_{\mathrm{BL}}$ |
| current | $I$ | uA | subscript names the device, e.g. $I_{\mathrm{R}}$ |
| conductance | $G$ | uS | $G$ is conductance, never a count |
| resistance | $R$ | MOhm | $R$ is resistance, never a count |
| capacitance | $C$ | fF | |
| temperature | $T$ | K | |
| energy | $E$ | fJ | |
| time / duration | $t$ | ns | e.g. $t_{\mathrm{WL}}$ |
| area | $A$ | um^2 | |
| leakage power | $P$ | uW | |

## Counts and geometry (dimensionless)

Use a symbol only when the count enters an equation; otherwise refer to it by code field (`col_num`, `group_num`, `data_num`).

| Quantity | Symbol | When to use a symbol |
|---|---|---|
| rows | $N_{\mathrm{row}}$ | enters equations (e.g. ideal rescale) — keep |
| columns | $N_{\mathrm{col}}$ | only when in an equation; otherwise write `col_num` |
| reference groups | $N_{\mathrm{group}}$ | only when in an equation; otherwise write `group_num` |
| digits per word | $D$ | enters the radix fold — keep |
| radix | $r$ | enters the radix fold — keep |

Pure structure counts (e.g. data per group) have no symbol; write the code field (`data_num`).

## Mathematical notation

| Use | Form |
|---|---|
| matrix / vector | bold, $\mathbf{W}$, $\mathbf{x}$ |
| index subscript | italic, $V_{\mathrm{BL},k}$ |
| node-name subscript | upright, $\mathrm{BL}$ |
| integer set / grid | $\mathcal{X}$ |
| floor / clamp | $\lfloor\cdot\rfloor$, $\operatorname{clamp}$ |
| named operator | $\operatorname{TIA}(\cdot)$ (use `\operatorname`, not `\mathrm`) |

## Units (ASCII)

`V`, `uA`, `uS`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`, `um`, `um^2` — matching the code `__` suffixes. Inside an equation a unit may be set in LaTeX, e.g. $\mu\mathrm{A}$. The set is closed, self-consistent, and chosen to keep magnitudes inside the well-conditioned range of `bfloat16` / `float32`. It is closed under the products that appear in circuit math, so no intermediate needs rescaling: $\mathrm{uA}\cdot\mathrm{V}=\mathrm{uW}$, $\mathrm{uW}\cdot\mathrm{ns}=\mathrm{fJ}$, $\mathrm{fF}\cdot\mathrm{V}^2=\mathrm{fJ}$, $\mathrm{MOhm}\cdot\mathrm{uA}=\mathrm{V}$, $\mathrm{uS}\cdot\mathrm{V}=\mathrm{uA}$. No unit conversion is permitted on any tensor-computation path; every runtime tensor is already in these units.

## Config units

Config fields are the human-interaction surface and follow established industrial conventions, even when those differ from the runtime units above (e.g. PDK mobility in $\mathrm{cm}^2/\mathrm{V}/\mathrm{s}$, SI constants in $\mathrm{J}/\mathrm{K}$ or $\mathrm{C}$). Each field carries its unit suffix in the name; no suffix means dimensionless. Because config units differ from the runtime units above, a config value is converted to runtime units before it reaches any tensor-computation path. The name-suffix grammar for both surfaces is in [contributing/naming_conventions](../contributing/naming_conventions.md); runtime dtype is in [contributing/code_style](../contributing/code_style.md).

## Physical constants

The canonical constants live in one place so device and analog modules pull them from a single source. Constant values are exact SI / CODATA-2018 figures; $T_{\mathrm{room}}$ is the default operating temperature used whenever no explicit $T$ (`T__K`) is supplied. The thermal voltage $V_T = k_B T / q$ is derived from the first two constants at the given temperature.

| Quantity | Symbol | Code | Value | Unit | Source |
|---|---|---|---|---|---|
| elementary charge | $q$ | `ELEM_CHARGE__C` | $1.602176634 \times 10^{-19}$ | C | Constant |
| Boltzmann constant | $k_B$ | `K_BOLTZMANN__J_per_K` | $1.380649 \times 10^{-23}$ | J/K | Constant |
| vacuum permittivity | $\varepsilon_0$ | `EPS_0__F_per_m` | $8.8541878128 \times 10^{-12}$ | F/m | Constant |
| room temperature | $T_{\mathrm{room}}$ | `T_ROOM__K` | $300.0$ | K | Constant |
| thermal voltage | $V_T$ | `thermal_voltage__V(T__K)` | $k_B T / q$ | V | Constant |

## Sign conventions

TODO (domain author): state the sign conventions, including the signed-code convention of the ADC and the differential readout.

## Noise-model conventions

Every subsystem's Noise section parameterises its non-idealities against the shared template below, so a noise source carries one spelling everywhere. A non-ideality is an additive or multiplicative perturbation drawn from a distribution whose spread $\sigma$ comes in one of two flavours.

### State-independent vs state-dependent

- **State-independent** — $\sigma$ is a config constant; the same distribution is sampled at every element. Use for noise whose magnitude does not track signal magnitude (e.g. comparator thermal noise, stuck-at faults).
- **State-dependent** — $\sigma$ is derived per element from the input tensor (or auxiliary tensors). Use for noise whose magnitude grows with the conductance state (e.g. programming variability, retention drift) or with the cell area (e.g. Pelgrom mismatch, $kT/C$ sampling).

### Per-source policy toggle

Each noise source is independently switchable, and a disabled source is a no-op that passes its input through unchanged.

### Pelgrom area-scaled mismatch

Pelgrom multiplicative mismatch is a state-dependent source whose per-element spread shrinks with device area. For an element of capacitance $C_k$ relative to a unit cell $C_{\mathrm{unit}}$,

$$\sigma_k = \sigma_{\mathrm{rel}} \cdot \sqrt{\frac{C_k}{C_{\mathrm{unit}}}},$$

so $\sigma_k \propto \sqrt{C_k / C_{\mathrm{unit}}}$. A positive floor clamps $\sigma_k$ from below, $\sigma_k \ge \sigma_{\mathrm{floor}} > 0$, so the spread never collapses to zero for large devices.
