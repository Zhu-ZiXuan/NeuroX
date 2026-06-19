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

The shared symbol $t$ (ns) is a circuit / compute-path duration. Retention / drift time is a separate quantity carried in seconds — a much larger timescale — and is not this ns compute-path $t$.

## Counts and geometry (dimensionless)

Use a symbol only when the count enters an equation; otherwise refer to it by code field (`col_num`, `group_num`, `slice_num`).

| Quantity | Symbol | When to use a symbol |
|---|---|---|
| rows | $N_{\mathrm{row}}$ | enters equations (e.g. ideal rescale) — keep |
| columns | $N_{\mathrm{col}}$ | only when in an equation; otherwise write `col_num` |
| reference groups | $N_{\mathrm{group}}$ | only when in an equation; otherwise write `group_num` |
| digits per slice (digit count) | $D$ | enters the radix fold — keep ($D$ = `digit_count` = the xbar `digit_count`) |
| digit radix | $r$ | enters the radix fold — keep ($r$ = `digit_radix`, the base of one cell's digit) |
| slice radix | $R$ | enters the radix-weighted shift-add; $R = r^{D}$ ($R$ = `slice_radix`, the positional ratio between adjacent slices) |
| weight-slice count | $S_w$ | enters the precision-slicing fold (weight side); config-given, not inferred |
| activation-slice count | $S_a$ | enters the precision-slicing fold (input side); config-given, not inferred |
| output-axis tile count | $T_r$ | matrix-tiling axis; rows of the transposed weight, $T_r = \lceil N / N_{\mathrm{col}} \rceil$ |
| contraction-axis tile count | $T_c$ | matrix-tiling axis; $T_c = \lceil K / N_{\mathrm{row}} \rceil$ |

These names follow the three value-domain levels: a **digit** (level 0, the integer symbol one xbar cell carries at digit radix $r$), a **slice** (level 1, a fixed-capacity positional piece = $D$ digits at radix $r$, with slice radix $R = r^{D}$), and a **value** (level 2, the role-neutral algorithm scalar — a weight on the weight side, an activation on the input side). Precision slicing ($S_w$, $S_a$) cuts a value into slices; matrix tiling ($T_r$, $T_c$) is the orthogonal, application-neutral axis that splits any matmul. The per-slice value range is computed from $D$ and $r$ and published by the xbar interface (the authority); the slice counts $S_w$, $S_a$ are config-given.

The slice radix $R$ is dimensionless and lives in this value-domain table; it is distinct from the resistance $R$ (MOhm) of the electrical table — context (radix fold vs circuit equation) keeps them apart.

Pure structure counts (e.g. slices per group) have no symbol; write the code field (`slice_num`).

## Mathematical notation

| Use | Form |
|---|---|
| matrix / vector | bold, $\mathbf{W}$, $\mathbf{x}$ |
| index subscript | italic, $V_{\mathrm{BL},k}$ |
| node-name subscript | upright, $\mathrm{BL}$ |
| integer set / grid | $\mathcal{X}$ |
| floor / clamp | $\lfloor\cdot\rfloor$, $\operatorname{clamp}$ |
| modulo | $a \bmod n$ is the non-negative (Euclidean) residue in $[0, n)$, e.g. $(-1) \bmod 4 = 3$ — the convention the signed two's-complement wrap formulas rely on |
| named operator | $\operatorname{TIA}(\cdot)$ (use `\operatorname`, not `\mathrm`) |

## Units (ASCII)

`V`, `uA`, `uS`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`, `um`, `um^2` — matching the code `__` suffixes. Inside an equation a unit may be set in LaTeX, e.g. $\mu\mathrm{A}$. The set is closed, self-consistent, and chosen to keep magnitudes inside the well-conditioned range of floating-point arithmetic (the magnitude sweet spot is similar across float types). It is closed under the products that appear in circuit math, so no intermediate needs rescaling: $\mathrm{uA}\cdot\mathrm{V}=\mathrm{uW}$, $\mathrm{uW}\cdot\mathrm{ns}=\mathrm{fJ}$, $\mathrm{fF}\cdot\mathrm{V}^2=\mathrm{fJ}$, $\mathrm{MOhm}\cdot\mathrm{uA}=\mathrm{V}$, $\mathrm{uS}\cdot\mathrm{V}=\mathrm{uA}$. No unit conversion is permitted on any tensor-computation path; every runtime tensor is already in these units.

## Config units

Config fields are the human-interaction surface and follow established industrial conventions, even when those differ from the runtime units above (e.g. PDK mobility in $\mathrm{cm}^2/\mathrm{V}/\mathrm{s}$, SI constants in $\mathrm{J}/\mathrm{K}$ or $\mathrm{C}$). Each field carries its unit suffix in the name; no suffix means dimensionless. Because config units differ from the runtime units above, a config value is converted to runtime units before it reaches any tensor-computation path. The name-suffix grammar for both surfaces is in [contributing/naming_conventions](../contributing/naming_conventions.md); runtime dtype is in [contributing/code_style](../contributing/code_style.md).

## Physical constants

The canonical constants live in one place so device and analog modules pull them from a single source. The elementary charge $q$ and Boltzmann constant $k_B$ are exact by SI definition (zero uncertainty); the vacuum permittivity $\varepsilon_0$ is a measured / derived quantity carrying a relative uncertainty of $\sim 1.6 \times 10^{-10}$, listed at its CODATA-2018 value. $T_{\mathrm{room}}$ is the default operating temperature used whenever no explicit $T$ (`T__K`) is supplied. The thermal voltage $V_T = k_B T / q$ is derived from the first two constants at the given temperature.

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
- **State-dependent** — $\sigma$ is derived per element from the input tensor (or auxiliary tensors). Use for noise whose magnitude tracks the conductance state (e.g. programming variability) or scales with (depends on) the cell area (e.g. Pelgrom mismatch, $kT/C$ sampling, both shrinking as $1/\sqrt{\mathrm{area}}$).

A retention drift is a separate, time-dependent / deterministic perturbation, not a stochastic $\sigma$-spread flavour: it is a time-only gain applied to the stored state (e.g. the power-law conductance drift $d(t) = (t/t_0)^{-\nu}$ of [device/rram](device/rram.md)), governed by device parameters and elapsed time rather than a sampled distribution.

### Per-source policy toggle

Each noise source is independently switchable, and a disabled source is a no-op that passes its input through unchanged.

### Pelgrom area-scaled mismatch

Pelgrom mismatch is a state-dependent source built on one principle: a parameter that is the area-average of spatially-uncorrelated microscopic fluctuations has a variance proportional to $1/\mathrm{area}$, so its relative spread shrinks as $1/\sqrt{\mathrm{area}}$ — larger devices match better. The canonical two-term Pelgrom law for a parameter difference $\Delta P$ between two devices is

$$\sigma^2(\Delta P) = \frac{A_P^2}{W L} + S_P^2\, D^2,$$

a local area term ($A_P$ over the gate-region product $W L$) plus a separate long-range gradient term ($S_P$ scaling the device separation $D$).

For a capacitor the source is multiplicative ($\sigma_{\mathrm{rel}}$), so the template is committed to the relative form, with $C_k$ the element capacitance and $C_{\mathrm{unit}}$ the unit cell:

$$\sigma\!\left(\frac{\Delta C_k}{C_k}\right) = \sigma_{\mathrm{rel}} \cdot \sqrt{\frac{C_{\mathrm{unit}}}{C_k}},$$

so the relative spread is $\propto 1/\sqrt{C_k}$ and shrinks with area. The equivalent absolute spread $\sigma(\Delta C_k) = \sigma_{\mathrm{rel}}\sqrt{C_{\mathrm{unit}}\, C_k}$ instead grows as $\sqrt{C_k}$; the two framings of the same $1/\mathrm{area}$ variance look opposite only because one is normalised by $C_k$ and the other is not.

The local area term vanishes monotonically as the device grows; there is no floor. The genuine non-vanishing large-area residual is the separate, area-independent long-range gradient term $S_P^2 D^2$, modelled as an additive variance contribution (not a clamp on $\sigma$). TODO (domain author): the concrete parameterisation of the distance term (the $S_P$ coefficient and the $D$ separation, and their config homes).

**Absolute vs relative framing.** Use the absolute spread for an intensive parameter — a threshold voltage adds directly, so $\sigma(\Delta V_{\mathrm{th}}) \propto 1/\sqrt{W L}$. Use the relative spread for a multiplicative parameter — a current factor $\beta$ or a capacitance $C$ perturbs its operand proportionally. It is the same $1/\mathrm{area}$ variance in both cases; the apparent area-direction differs only by this framing.
