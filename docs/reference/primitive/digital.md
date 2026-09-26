# Digital circuits

## Shared model

Digital circuits produce signed integer results with output-width wrapping. Area, leakage, and evaluation energy characterize the selected implementation's width and spatial input capacity; the enclosing unit schedules operations.

For output width $w$,

$$\operatorname{wrap}_w(z)=\left[(z+2^{w-1})\bmod 2^w\right]-2^{w-1}.$$

A reduction of operands $x_i$, positional weights $a_i$, and enables $\eta_i\in\{0,1\}$ gives

$$y=\operatorname{wrap}_w\left(\sum_{i=0}^{n-1}\eta_i a_i x_i\right).$$

Disabled operands contribute zero and incur no evaluation energy; enabled zeros incur energy. An empty reduction returns zero. Wrapping each intermediate sum is equivalent to wrapping the complete sum once.

## Circuit models

| Circuit | Structure | Positional weight $a_i$ |
| --- | --- | --- |
| Accumulator | Feedback register updated by a temporal stream | $1$ |
| Radix accumulator | Feedback register updated by a positional digit stream | $r^i$ |
| Summator | Parallel adder tree | $1$ |
| Radix summator | Parallel positional-sum tree | $r^i$ |

Accumulators start at zero and update once per arrival. Parallel trees reduce the complete operand set; their characterized spatial fan-in and PPA must match its extent.

### Adder

A two-input adder produces

$$y=\operatorname{wrap}_w\bigl(\eta(a+b)\bigr).$$

One enabled output incurs one complete addition's evaluation cost, including a zero result. A disabled operation returns zero without dynamic energy.

### Radix accumulator

The register receives $D$ digits in least-significant-first order, with $n=D$ and $r\geq1$. Radix one gives equal-weight accumulation; larger radices reconstruct positional digits.

### Radix summator

The tree combines $D$ digits under the same positional convention. Every operand is enabled unless gated. For both radix circuits, powers of two correspond to binary shifts; the model applies configured evaluation costs to every accepted integer radix.

## Costs

Each enabled reduction operand incurs one evaluation; an enabled two-input addition incurs one evaluation for the complete operation. With evaluation count $N_{\mathrm{op}}$ and physical instance count $N_{\mathrm{inst}}$,

$$E=N_{\mathrm{op}}E_{\mathrm{op}},\qquad A=N_{\mathrm{inst}}A_{\mathrm{inst}},\qquad P=N_{\mathrm{inst}}P_{\mathrm{inst}}.$$

Temporal reuse counts static costs once and bills every enabled update.

## Parameters

| Parameter | Meaning | Unit | Constraint | [Source](../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| $w$ | signed output width | bit | $w\geq1$ | Design |
| $E_{\mathrm{op}}$ | energy per enabled evaluation | fJ | $E_{\mathrm{op}}\geq0$ | Design |
| $A_{\mathrm{inst}}$ | area per physical instance | um^2 | $A_{\mathrm{inst}}\geq0$ | Design |
| $P_{\mathrm{inst}}$ | leakage per physical instance | uW | $P_{\mathrm{inst}}\geq0$ | Design |

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $z,y$ | unwrapped value and output | — | arithmetic intermediate and result |
| $w$ | signed output width | bit | `bit_width` |
| $a,b,x_i$ | adder inputs and reduction operands | — | `a`, `b`, `x` |
| $a_i$ | positional weight | — | radix weighting |
| $\eta,\eta_i$ | operation and operand enables | — | `enable` |
| $i,n,D$ | operand index, operand count, and digit count | — | reduction axis |
| $r$ | digit radix | — | `radix` |
| $N_{\mathrm{op}},N_{\mathrm{inst}}$ | enabled evaluation and physical instance counts | — | evaluation count, `inst_count` |
| $E,A,P$ | dynamic energy, local area, and local leakage | fJ, um^2, uW | dynamic energy, `area__um2`, `leakage__uW` |
| $E_{\mathrm{op}}$ | energy per evaluation | fJ | `energy_per_op__fJ` |
| $A_{\mathrm{inst}},P_{\mathrm{inst}}$ | per-instance area and leakage | um^2, uW | `area_per_inst__um2`, `leakage_per_inst__uW` |

## Assumptions, scope & validity

Arithmetic is exact before wrapping when the chosen integer storage represents all operands and intermediates. The characterized implementation meets timing at the configured period. The model omits data-dependent propagation delay, switching activity, clock skew, and timing violations.

Changing frequency alone supplies no area or leakage scaling law. A slower clock on the same implementation preserves area and, at unchanged voltage and temperature, characterized leakage. Geometry changes require matching characterization data.

## Validation

Physical characterization of the supplied PPA values is not provided here.
