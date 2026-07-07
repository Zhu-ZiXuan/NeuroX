# Notation & conventions

The single home of the notation policy — the character whitelist, where a formula lives, and the shared symbol vocabulary. Reference, Internals, and in-code docstrings and comments all follow it.

Symbols **reused across subsystems** are pinned so one physical quantity keeps one symbol everywhere. Introduce a symbol only when it appears in an equation; a quantity that only labels a structure or count is written by its code field, not a symbol.

## Electrical and physical quantities

| Quantity | Symbol | Unit | Notes |
|---|---|---|---|
| voltage | $V$ | V | subscript names the node, e.g. $V_{\mathrm{BL}}$ |
| current | $I$ | uA | subscript names the device, e.g. $I_{\mathrm{R}}$ |
| conductance | $G$ | uS | |
| resistance | $R$ | MOhm | $R$ is resistance, never a count |
| capacitance | $C$ | fF | |
| temperature | $T$ | K | |
| energy | $E$ | fJ | |
| time / duration | $t$ | ns | e.g. $t_{\mathrm{WL}}$ |
| area | $A$ | um^2 | |
| power | $P$ | uW | |

## Counts and geometry (dimensionless)

Use a symbol only when the count enters an equation; otherwise refer to it by code field (`col_num`, `group_num`, `slice_num`).

| Quantity | Symbol | Code field | When to use a symbol |
|---|---|---|---|
| rows | $N_{\mathrm{row}}$ | `row_num` | enters equations (e.g. ideal rescale) |
| columns | $N_{\mathrm{col}}$ | `col_num` | only when in an equation |
| reference groups | $N_{\mathrm{group}}$ | `group_num` | only when in an equation |
| digits per slice (digit count) | $D$ | `digit_count` | enters the radix fold |
| digit radix | $r$ | `digit_radix` | enters the radix fold (base of one cell's digit) |
| slice radix | $R$ | `slice_radix` | enters the radix-weighted shift-add; $R = r^{D}$ (positional ratio between adjacent slices) |
| weight-slice count | $S_w$ | `w_slice_num` | enters the precision-slicing fold (weight side) |
| activation-slice count | $S_a$ | `x_slice_num` | enters the precision-slicing fold (input side) |
| output-axis tile count | $T_r$ | — | matrix-tiling axis; rows of the transposed weight, $T_r = \lceil N / N_{\mathrm{col}} \rceil$ |
| contraction-axis tile count | $T_c$ | — | matrix-tiling axis; $T_c = \lceil K / N_{\mathrm{row}} \rceil$ |

These names pin the symbols for the three value-domain levels — digit, slice, value — whose semantics are defined in [glossary §Value domain and slicing](glossary.md#value-domain-and-slicing). Precision slicing ($S_w$, $S_a$) cuts a value into slices; matrix tiling ($T_r$, $T_c$) is the orthogonal, application-neutral axis that splits any matmul. The per-slice value range is computed from $D$ and $r$ and published by the xbar interface (the authority).

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
| modulo | $a \bmod n$ is the non-negative (Euclidean) residue in $[0, n)$, e.g. $(-1) \bmod 4 = 3$ |
| named operator | $\operatorname{ADC}(\cdot)$ |

Named operators use `\operatorname{}`; reserve `\mathrm{}` for upright labels and subscripts.

## Character whitelist

One whitelist governs every layer — Markdown prose, docstrings, and comments share the **same** permitted non-ASCII set, written as the raw glyph. Inside a real formula a glyph takes its LaTeX form instead (last column, used within `$...$`); everything outside the set is ASCII.

| Permitted | Use | In a formula |
|---|---|---|
| — | aside / parenthetical | prose only |
| § | section reference (`§Parameters`) | prose only |
| × | multiplication in running text (`5×`, `64×64`, `batch×inst`) | `\times` |
| · ± ≈ ≤ ≥ ≠ | relation / operator | `\cdot` `\pm` `\approx` `\leq` `\geq` `\neq` |
| → ← ↔ | flow / mapping / correspondence | `\to` `\gets` `\leftrightarrow` |
| α β γ … ω | math variable (σ std-dev, μ mean) | `\sigma`, `\mu`, … |
| Γ Δ Θ Λ Ξ Π Σ Φ Ψ Ω | math variable (Greek capitals distinct from Latin) | `\Delta`, … |
| ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁿ | power of a math variable (σ²) | `^{2}` |
| ∂ | partial derivative (∂I/∂V) | `\partial` |

A lone whitelisted symbol may stay raw in prose; a symbol that is part of a formula takes the LaTeX form. Subscripts are ASCII `_` in every layer (`σ_k` in a docstring, `\sigma_k` in a formula), never a Unicode subscript. Identifiers stay ASCII regardless — a variable is `sigma` in code, σ in its docstring.

A non-whitelisted symbol — $\sum$, $\int$, $\prod$, $\nabla$, $\in$, $\infty$, $\propto$, $\perp$, $\sqrt{\;}$, and every other non-ASCII glyph — appears only inside a real formula, which lives in md-LaTeX.

Outside the whitelist entirely, ASCII everywhere: identifiers, data and protocol string literals, and physical units with their SI prefixes (`Ohm`, `u`, `uA`, `deg`, `um^2`). Units are user-locked (§Units and naming); the whitelist governs prose and notation, never these.

### Same glyph, two roles

One Greek glyph can be a math variable or a physical unit / SI prefix, written differently. As a **math variable** (mean μ, angular frequency ω) it follows the table — raw glyph in docstrings and comments, LaTeX in a formula. As a **physical unit or SI prefix** (ohm, micro) it is the ASCII name everywhere — `MOhm`, `uA`, `um` — never the glyph, never `$\Omega$` or `$\mu\mathrm{A}$`. A raw μ or Ω is therefore always the math variable, never a unit.

## Notation by string class

A string's notation follows who consumes it and whether a renderer sits between the source and that reader.

| Class | The reader sees | Members | Notation |
|---|---|---|---|
| Executed as code | a machine parses, compares, or executes it | identifiers (variable/function/class/module/parameter names); dict and set keys; registry and enum keys; data or protocol string literals (config field names, TOML/JSON keys, serialization tags); einsum subscripts; regexes; format-spec placeholders; code and file paths; CLI flag and env-var names | ASCII only |
| Shown raw to a human | the source text itself, unrendered | inline comments; docstrings; log and `print` output; exception, warning, and assert messages; CLI help; progress text | raw whitelisted unicode; never LaTeX; simple inline notation only |
| Rendered for a human | a renderer produces the visual | Markdown docs; notebook Markdown cells; any string fed to a LaTeX, HTML, or markup renderer | a formula's symbols use the renderer's LaTeX (`$\sigma$`); units are ASCII |

A class-1 token embedded in a class-2 or class-3 string stays ASCII: an exception message naming a field, a rendered label carrying a config key.

A docstring is class 2. Its primary readers see it raw — `help()`, an IDE tooltip, `__doc__`, the source, a diff — so it carries raw unicode and never LaTeX.

## Placement — where a formula lives

A complex, multi-term formula lives **only** in a Markdown Reference or Internals doc, written as LaTeX. A docstring or comment carries at most simple inline notation — a lone symbol (σ), a power of a variable (σ²), a short inline expression, an inline derivative (∂I/∂V) — and otherwise points to the md spec. A superscript power is for a math variable (σ², V_BL²); a physical unit raised to a power stays ASCII (`cm^2`, `um^2`). A docstring never hosts a multi-term derivation; it names the quantity and links the equation to its md home.

## Units and naming

This section defines the unit set and naming grammar.

### Runtime units (ASCII)

`V`, `uA`, `uS`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`, `um`, `um^2` — matching the code `__` suffixes. The set is closed and self-consistent under the products that appear in circuit math, so no intermediate needs rescaling. No unit conversion is permitted on any tensor-computation path; every runtime tensor is already in these units.

### Config units

Config fields are the human-interaction surface and follow established industrial conventions, even when those differ from the runtime units above (e.g. PDK mobility in $\mathrm{cm}^2/\mathrm{V}/\mathrm{s}$, SI constants in $\mathrm{J}/\mathrm{K}$ or $\mathrm{C}$). Each field carries its unit suffix in the name.

### Name-suffix grammar

Every variable or config field carrying a physical quantity uses `<name>__<unit>`, even when the surrounding text already states the unit; a dimensionless quantity has no suffix.

- Separator: a double underscore `__` joins the name to the unit.
- Unit case follows the physical standard.
- Multiplication is implicit, joining adjacent unit tokens with `_`, e.g. `A_vt__mV_um`.
- Division uses `_per_`, e.g. `mu0__cm2_per_V_s`.

## Physical constants

The canonical constants live in one place so device and analog modules pull them from a single source. The elementary charge $q$ and Boltzmann constant $k_B$ are exact by SI definition (zero uncertainty); the vacuum permittivity $\varepsilon_0$ is a measured / derived quantity carrying a relative uncertainty of $\sim 1.6 \times 10^{-10}$, listed at its CODATA-2018 value. $T_{\mathrm{room}}$ is the default operating temperature used whenever no explicit $T$ (`T__K`) is supplied. The thermal voltage $V_T = k_B T / q$ is derived from the first two constants at the given temperature.

| Quantity | Symbol | Code | Value | Unit | Source |
|---|---|---|---|---|---|
| elementary charge | $q$ | `ELEM_CHARGE__C` | $1.602176634 \times 10^{-19}$ | C | Constant |
| Boltzmann constant | $k_B$ | `K_BOLTZMANN__J_per_K` | $1.380649 \times 10^{-23}$ | J/K | Constant |
| vacuum permittivity | $\varepsilon_0$ | `EPS_0__F_per_m` | $8.8541878128 \times 10^{-12}$ | F/m | CODATA-2018 |
| room temperature | $T_{\mathrm{room}}$ | `T_ROOM__K` | $300.0$ | K | Default |
| thermal voltage | $V_T$ | `thermal_voltage__V(T__K)` | $k_B T / q$ | V | Constant-derived |
