# Notation and units

Shared symbols and unit scales used by the model reference and public interfaces.

Symbols **reused across subsystems** are pinned so one physical quantity keeps one symbol everywhere. Introduce a symbol only when it appears in an equation; a quantity that only labels a structure or count is written by its code field, not a symbol.

## Electrical and physical quantities

| Quantity | Symbol | Unit | Notes |
| --- | --- | --- | --- |
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
| --- | --- | --- | --- |
| rows | $N_{\mathrm{row}}$ | `row_num` | enters equations (e.g. ideal rescale) |
| columns | $N_{\mathrm{col}}$ | `col_num` | only when in an equation |
| reference groups | $N_{\mathrm{group}}$ | `group_num` | only when in an equation |
| digits per slice (digit count) | $D$ | `digit_count` | enters the radix fold |
| digit radix | $r$ | `digit_radix` | enters the radix fold (base of one cell's digit) |
| slice radix | $R$ | `slice_radix` | enters the radix-weighted shift-add; $R = r^{D}$ (positional ratio between adjacent slices) |
| weight-slice count | $S_w$ | `w_slice_num` | enters the precision-slicing fold (weight side) |
| activation-slice count | $S_x$ | `x_slice_num` | enters the precision-slicing fold (input side) |
| contraction-axis tile count | $T_c$ | — | defined by the selected mapping |

These names pin the symbols for the three value-domain levels — digit, slice, value — whose semantics are defined in [glossary §Value domain and slicing](glossary.md#value-domain-and-slicing). Precision slicing ($S_w$, $S_x$) cuts a value into slices; matrix tiling is the orthogonal, application-neutral axis that splits any matmul. The per-slice value range is computed from $D$ and $r$ and published by the macro interface.

The slice radix $R$ is dimensionless and lives in this value-domain table; it is distinct from the resistance $R$ (MOhm) of the electrical table — context (radix fold vs circuit equation) keeps them apart.

## Mathematical notation

| Use | Form |
| --- | --- |
| matrix / vector | bold, $\mathbf{W}$, $\mathbf{x}$ |
| index subscript | italic, $V_{\mathrm{BL},k}$ |
| node-name subscript | upright, $\mathrm{BL}$ |
| integer set / grid | $\mathcal{X}$ |
| floor / clamp | $\lfloor\cdot\rfloor$, $\operatorname{clamp}$ |
| modulo | $a \bmod n$ is the non-negative (Euclidean) residue in $[0, n)$, e.g. $(-1) \bmod 4 = 3$ |
| named operator | $\operatorname{ADC}(\cdot)$ |

Named operators use `\operatorname{}`; reserve `\mathrm{}` for upright labels and subscripts.

## Units and naming

This section defines the unit set and naming grammar.

### Standard unit atoms and expressions (ASCII)

The standard atoms are `V`, `uA`, `uS`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`, and `um`. A suffix composed solely from these atoms is standard too: `_` joins a product, `_per_` introduces one denominator, a leading `per_` means a reciprocal, and a trailing integer is a power. Thus `fF_per_um2`, `uA_per_V2`, and `per_V` need no special ruling.

The spelling is not algebraically normalized. An author may retain the expression that best exposes the calculation, so `V_per_uA` need not be rewritten as `MOhm`, and `V_uA_ns` need not be rewritten as `fJ`. The standard atoms still fix the numerical scales: no unit conversion is permitted on a tensor-computation path.

### Nonstandard unit expressions

Config fields may follow an established industrial convention, while physical constants and validation reports may retain an external source's unit. Examples include PDK mobility in $\mathrm{cm}^2/\mathrm{V}/\mathrm{s}$ and SI constants in $\mathrm{J}/\mathrm{K}$ or $\mathrm{C}$. Each code binding carries its unit suffix in the name; a nonstandard expression is admitted only for that exact binding.

### Name-suffix grammar

Every identifier naming a physical quantity uses `<name>__<unit>`, even when the surrounding text already states the unit — a parameter, a dataclass or config field, and a local alike. A function or property whose return value is one physical quantity is named like the variable that would hold it, the quantity first and the unit suffix last (`a__V()`); a call returning several quantities keeps a bare name and its elements take their suffixes at the unpack (`a__V, b__uA = ab()`). A dimensionless quantity has no suffix.

- Separator: exactly one interior double underscore `__` joins the name to the unit. Leading or trailing dunders keep their Python meaning; another interior `__` is invalid.
- Unit case follows the physical standard.
- Multiplication is implicit, joining adjacent unit tokens with `_`, e.g. `A_vt__mV_um`.
- Division uses one `_per_`, e.g. `mu0__cm2_per_V_s`; `per_V` denotes a reciprocal.
- A trailing integer denotes a power, e.g. `V2` or `um2`.

The rules test reads Python definitions and bindings. It does not infer physical meaning, inspect prose, or treat an identifier-looking comment, docstring, or string as code.

### Derivative identifiers

A quantity that is a derivative is named `d<y>_d<x>__<unit>`, all lowercase. Each of `<y>` and `<x>` is the quantity's own name with its internal underscores removed, so the run-together name leaves `_d` as the one separator in the identifier. Its suffix follows the same expression grammar as every other unit; the table uses the compact standard atom where one is convenient.

| Derivative | Identifier | Unit |
| --- | --- | --- |
| ∂a/∂b, a in V, b in uA | `da_db__MOhm` | MOhm |
| ∂a/∂b, a in uA, b in V | `da_db__uS` | uS |
| ∂a_out/∂b, a_out in V, b in uA | `daout_db__MOhm` | MOhm |

This is the identifier register only; a docstring or comment naming the same quantity in prose writes it as ∂a/∂b under the source-text character convention, and a Markdown formula writes `\partial`.

### Runtime scales

The standard atoms establish the runtime scales, not one mandatory algebraic spelling for each dimension. Energy, for example, uses the fJ scale; `fJ` and the unreduced standard expression `V_uA_ns` have the same numerical scale. A differently scaled atom such as `pJ` is nonstandard and requires an exact code-binding ruling. An external source may state another scale in prose without turning that prose into a code identifier.

## Physical laws and constants

[Physics](../reference/primitive/physics.md) owns the charge and energy laws and physical constants. [PPA accounting](../system_design/ppa_accounting.md) defines the modeled event and powered-window boundaries.
