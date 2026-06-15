# Shift-Adder

## Summary

The shift-adder is the radix-weighted positional-sum primitive that recombines a multi-digit integer representation into a single value: it folds a digit axis with positional radix weights, wraps the result to a signed register, and optionally adds a running partial sum. It is the digital counterpart of the radix fold a multi-digit datapath needs after the per-digit results are produced. It is an exact digital block; the only modelled physical content is its behavioural PPA cost.

## Physical model

The block abstracts a shift-and-add reduction unit: each digit position is scaled by the corresponding power of the radix (a shift in a power-of-two radix) and summed into a fixed-width signed register, with two's-complement modular wrap on overflow. An optional partial-sum input is added after the wrap, letting the unit chain across calls. The model is behavioural — no per-stage adder timing — so the only physical quantities exposed are the PPA cost terms.

## Governing equations

Let the digit axis have length $D$, indexed $i = 0, \dots, D-1$, with radix $r$. The positional weight vector is $(1, r, r^2, \dots, r^{\,D-1})$. With register width $w$, half-range $h = 2^{\,w-1}$ and full-range $f = 2^{\,w}$, the radix-weighted sum is folded into the signed register range and then offset by the optional partial sum $p$,

$$y = \left[\left(\sum_{i=0}^{D-1} r^{\,i}\, x_i + h\right) \bmod f\right] - h + p,$$

where $p$ is added only when supplied (otherwise $p = 0$, and it is added after the wrap, so it can itself carry the running total past the wrapped range). The sum runs over the single digit axis; all other axes are independent instances.

## Numerical method

N/A - exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A - exact digital function; the only non-infinite-precision effect is the deterministic modular wrap of the register, captured in §Governing equations. No static mismatch, no per-call randomness.

## PPA cost model

Per reduced output element the block dissipates a fixed dynamic energy $E_{\mathrm{op}}$; one output element is one shift-add evaluation. The serial-op count of a call is the number of output elements per fabricated instance,

$$n_{\mathrm{serial}} = \max\!\left(1,\ \left\lfloor \frac{\operatorname{numel}(y)}{\max(N_{\mathrm{inst}}, 1)} \right\rfloor\right),$$

where $\operatorname{numel}(y)$ already excludes the reduced digit axis. The call's dynamic energy is $E_{\mathrm{op}}$ per output element and its latency is $t = t_{\mathrm{op}}\, n_{\mathrm{serial}}$. Static area and leakage are the inherited per-instance terms scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ and any dependence on the digit count $D$ or radix $r$; the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `bit_width` | signed output register width | — | Design |
| `energy_per_op__fJ` | dynamic energy per output element | fJ | Design |
| `latency_per_op__ns` | latency per output element | ns | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | Design |

The radix $r$ and the optional partial sum $p$ are runtime call arguments, not configuration. Provenance terms: [parameter_provenance](../parameter_provenance.md). File-level schema: `neurox/digital/shift_adder.py` (`ShiftAdderConfig`).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x$ | integer digit tensor (runtime input) | — | `x` |
| $x_i$ | digit at position $i$ along the digit axis | — | slice of `x` |
| $D$ | digit count (length of the digit axis) | — | `x.size(dim)` |
| $r$ | positional radix | — | `scale` |
| $p$ | optional partial-sum offset (runtime input) | — | `init_val` |
| $y$ | recombined, wrapped, offset output | — | return of `operate` |
| $w$ | signed output register width | — | `bit_width` |
| $E_{\mathrm{op}}$ | dynamic energy per output element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | latency per output element | ns | `latency_per_op__ns` |
| $n_{\mathrm{serial}}$ | serial-op count of a call | — | `serial_op_count` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- The output register wraps in two's-complement on the radix-weighted sum; the optional partial sum $p$ is added after the wrap and is not itself bounded by the register.
- The cost model is behavioural and per-op flat: energy and latency scale only with the output-element count, not with the digit count $D$, the radix $r$, or operand magnitude.

TODO (domain author): the validity range of the flat per-op cost (whether $t_{\mathrm{op}}$ should grow with $D$ for a serial shift-add), and any conditions under which the post-wrap partial-sum semantics is a modelling error rather than intended behaviour.

## Validation

TODO (domain author): evidence that the radix-fold function and the PPA accounting match the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the shift-and-add recombination architecture and the PPA basis.

---

- **Internals**: [shift_adder internals](../../internals/digital/shift_adder.md)
- **Validation**: TODO - validation artefact not yet written
- **Configuration**: `neurox/digital/shift_adder.py` (`ShiftAdderConfig`)
- **Decisions**: N/A — no ADR governs this module.
