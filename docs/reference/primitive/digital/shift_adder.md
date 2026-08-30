# Shift-adder

The shift-adder is the radix-weighted positional-sum primitive that recombines a multi-digit integer representation into a single value: it folds a digit axis with positional radix weights, wraps the result to a signed register, and adds a partial-sum offset. It is an exact digital block; the only modelled physical content is its behavioural PPA cost.

## Physical model

The block abstracts a shift-and-add reduction unit: each digit position is scaled by the corresponding power of the radix and summed into a fixed-width signed register, with two's-complement modular wrap on overflow. A partial-sum offset is added after the wrap. The model is behavioural — no per-stage adder timing — so the only physical quantities exposed are the PPA cost terms.

## Governing equations

Let the digit axis have length $D$, indexed $i = 0, \dots, D-1$, with radix $r$. The positional weight vector is $(1, r, r^2, \dots, r^{\,D-1})$. With register width $w$, half-range $h = 2^{\,w-1}$ and full-range $f = 2^{\,w}$, the radix-weighted sum is folded into the signed register range and then offset by the partial sum $p$,

$$y = \left[\left(\sum_{i=0}^{D-1} r^{\,i}\, x_i + h\right) \bmod f\right] - h + p.$$

The offset $p$ is added after the wrap, so $y$ need not lie in the register's signed range. The sum runs over the single digit axis; all other axes are independent instances.

## Numerical method

N/A — exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A — exact digital function; the only non-infinite-precision effect is the deterministic modular wrap of the register, captured in §Governing equations. No static mismatch, no per-call randomness.

## PPA cost model

Per operand element folded into the sum the block dissipates a fixed dynamic energy $E_{\mathrm{op}}$; one digit leg of one output is one shift-and-add evaluation, so work scales with the input-element count and not with the number of results. The digit legs are weighted and summed in one pass, so the digit axis is spatial and the duration is the flat positional-sum window $t = t_{\mathrm{op}}$. Dynamic energy is total work, independent of how the operands distribute across instances,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(x),$$

where $\operatorname{numel}(x)$ includes the digit axis, so a fold over twice as many digits costs twice as much. The partial sum $p$ preloads the destination register and adds no evaluation of its own. An empty call, $\operatorname{numel}(x) = 0$, costs zero energy. Static area and leakage are the inherited per-instance terms scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ and any dependence on the digit count $D$ or radix $r$; the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `bit_width` | signed output register width | — | $> 0$ | Design |
| `scale` | positional radix $r$ (init argument) | — | $\geq 2$ | Design |
| `digit_count` | number of positional digits $D$ (init argument) | — | $\geq 1$ | Design |
| `energy_per_op__fJ` | dynamic energy per operand element folded into the sum | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | positional-sum window of one shift-add | ns | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | $\geq 0$ | Design |

The radix and digit count are bound when the physical block is constructed; the digit axis and partial sum remain runtime call arguments. Provenance terms: [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $x$ | integer digit tensor (runtime input) | — | `x` |
| $x_i$ | digit at position $i$ along the digit axis | — | slice of `x` |
| $D$ | digit count (length of the digit axis) | — | `digit_count` |
| $r$ | digit radix | — | `scale` (init argument) |
| $p$ | partial-sum offset (runtime input) | — | `init_val` |
| $y$ | recombined, wrapped, offset output | — | return of `shift_add` |
| $w$ | signed output register width | — | `bit_width` |
| $E_{\mathrm{op}}$ | dynamic energy per operand element folded into the sum | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | positional-sum window of one shift-add | ns | `latency_per_op__ns` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- The output register wraps in two's-complement on the radix-weighted sum; the partial sum $p$ is added after the wrap and is not itself bounded by the register.
- The cost model is behavioural and per-op flat: energy scales only with the input-element count, hence linearly in the digit count $D$, and the duration not at all, neither varying with $D$, the radix $r$, or operand magnitude.

TODO (domain author): the validity range of the flat per-op cost (whether $t_{\mathrm{op}}$ should grow with $D$ for a serial shift-add), and any conditions under which the post-wrap partial-sum semantics is a modelling error rather than intended behaviour.

## Validation

TODO (domain author): evidence that the radix-fold function and the PPA accounting match the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the shift-and-add recombination architecture and the PPA basis.
