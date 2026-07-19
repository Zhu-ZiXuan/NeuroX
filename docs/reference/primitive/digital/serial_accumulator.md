# Serial accumulator

The serial accumulator reduces an integer tensor along one time-serial axis into a single fixed-width signed register, folding each arriving operand into the running total one accumulate op at a time. Its integer function is identical to the [accumulator](accumulator.md); it differs only in the billing law — it bills per arriving input element, where the accumulator bills per reduced output element.

## Physical model

The model realizes a fixed-width signed accumulator register of $w$ bits fed by a time-serial operand stream: the reduced axis holds successive arrivals on one physical register, and each arrival triggers one accumulate op. On overflow the register wraps in two's-complement rather than clamping. It is behavioural and integer-exact — not a gate-level model — so the only physical quantities it exposes are the PPA cost terms.

## Governing equations

The block sums input $x$ along the reduction axis and folds the result into the signed range of a $w$-bit register by two's-complement modular wrap. With register width $w$, half-range $h = 2^{\,w-1}$ and full-range $f = 2^{\,w}$,

$$y = \left[\left(\sum_{k} x_k + h\right) \bmod f\right] - h,$$

which maps any integer sum into the signed interval $[-2^{\,w-1},\ 2^{\,w-1}-1]$. The sum runs over the single reduced axis; because intermediate partial sums also live in the wrapping register, the single modular wrap of the final sum is exact for the composed per-arrival wraps.

## Numerical method

N/A — exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A — exact digital function; the only non-infinite-precision effect is the deterministic modular wrap of the output register (§Governing equations). No static mismatch, no per-call randomness.

## PPA cost model

Each arriving operand is accumulated once, so the block dissipates a fixed dynamic energy $E_{\mathrm{op}}$ per input element and total work scales with the input-element count,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(x),$$

where $\operatorname{numel}(x)$ includes the reduced (time-serial) axis. Latency is set by the busiest instance: the serial-op count of a call is the number of input elements on the instance carrying the most work,

$$n_{\mathrm{serial}} = \left\lceil \frac{\operatorname{numel}(x)}{\max(N_{\mathrm{inst}}, 1)} \right\rceil,$$

so its latency is

$$t = t_{\mathrm{op}}\, n_{\mathrm{serial}}.$$

An empty call, $\operatorname{numel}(x) = 0$, costs zero latency and zero energy. Static area and leakage are the per-instance terms $A_{\mathrm{inst}}$ and $P_{\mathrm{inst}}$ scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ (bit-width scaling, technology node); the source docs give only the accounting form, not the values.

## Parameters

The block reuses the accumulator's configuration; the per-op terms are re-read against the input-element count.

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `bit_width` | signed output register width | — | $\geq 1$ | Design |
| `energy_per_op__fJ` | dynamic energy per input element | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | latency per input element | ns | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x$ | integer input tensor (runtime input) | — | `x` |
| $y$ | modular-wrapped reduced output | — | return of `operate` |
| $w$ | signed output register width | — | `bit_width` |
| $E_{\mathrm{op}}$ | dynamic energy per input element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | latency per input element | ns | `latency_per_op__ns` |
| $n_{\mathrm{serial}}$ | serial-op count of a call | — | `serial_op_count` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- The output register wraps in two's-complement and does not saturate; a sum exceeding the range silently aliases.
- The cost model is behavioural and per-op flat: energy and latency scale only with the input-element count, not with operand magnitude, bit toggling, or carry depth.
- The billing law assumes the reduced axis is genuinely time-serial on one register per instance; a reduction realized as a parallel adder tree is the accumulator's per-output billing instead.

TODO (domain author): the validity range of the flat per-op cost (bit-width regimes, the point at which carry-tree depth makes $t_{\mathrm{op}}$ bit-width-dependent), and any conditions under which modular wrap is a modelling error rather than the intended hardware behaviour.

## Validation

TODO (domain author): evidence that the modular-wrap function and the PPA accounting match the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the serial accumulate-register architecture and the PPA basis.

---

- **Internals**: [serial_accumulator internals](../../../internals/primitive/digital/serial_accumulator.md)
- **Validation**: TODO - validation artefact not yet written
- **Configuration**: `neurox/primitive/digital/accumulator.py` (`AccumulatorConfig`, reused)
