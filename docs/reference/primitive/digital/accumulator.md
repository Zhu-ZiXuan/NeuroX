# Accumulator

The accumulator reduces an integer tensor along one axis into a single fixed-width signed register. Summation along that axis is exact; the register's finite width, which wraps rather than saturates on overflow, is the model's only departure from infinite-precision integer arithmetic.

## Physical model

The model realizes a fixed-width signed accumulator register of $w$ bits: it sums the input along the reduction axis into that register, and on overflow the register wraps in two's-complement rather than clamping. It is behavioural and integer-exact — not a gate-level model — so the only physical quantities it exposes are the PPA cost terms.

## Governing equations

The accumulator sums input $x$ along the reduction axis and folds the result into the signed range of a $w$-bit register by two's-complement modular wrap. With register width $w$, half-range $h = 2^{\,w-1}$ and full-range $f = 2^{\,w}$,

$$y = \left[\left(\sum_{k} x_k + h\right) \bmod f\right] - h,$$

which maps any integer sum into the signed interval $[-2^{\,w-1},\ 2^{\,w-1}-1]$. The sum runs over the single reduced axis; all other axes are independent instances.

## Numerical method

N/A — exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A — exact digital function; the only non-infinite-precision effect is the deterministic modular wrap of the output register (§Governing equations). No static mismatch, no per-call randomness.

## PPA cost model

Per reduced output element the block dissipates a fixed dynamic energy $E_{\mathrm{op}}$, so work scales with the output-element count. Latency is set by the busiest instance: the serial-op count of a call is the number of output elements on the instance carrying the most work,

$$n_{\mathrm{serial}} = \left\lceil \frac{\operatorname{numel}(y)}{\max(N_{\mathrm{inst}}, 1)} \right\rceil,$$

where $\operatorname{numel}(y)$ already excludes the reduced axis, so its latency is

$$t = t_{\mathrm{op}}\, n_{\mathrm{serial}}.$$

Dynamic energy is total work, independent of how the outputs distribute across instances,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(y).$$

An empty call, $\operatorname{numel}(y) = 0$, costs zero latency and zero energy. Static area and leakage are the per-instance terms $A_{\mathrm{inst}}$ and $P_{\mathrm{inst}}$ scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ (bit-width scaling, technology node); the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `bit_width` | signed output register width | — | $\geq 1$ | Design |
| `energy_per_op__fJ` | dynamic energy per output element | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | latency per output element | ns | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x$ | integer input tensor (runtime input) | — | `x` |
| $y$ | modular-wrapped reduced output | — | return of `operate` |
| $w$ | signed output register width | — | `bit_width` |
| $E_{\mathrm{op}}$ | dynamic energy per output element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | latency per output element | ns | `latency_per_op__ns` |
| $n_{\mathrm{serial}}$ | serial-op count of a call | — | `serial_op_count` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- The output register wraps in two's-complement and does not saturate; a sum exceeding the range silently aliases.
- The cost model is behavioural and per-op flat: energy and latency scale only with the output-element count, not with operand magnitude, bit toggling, or carry depth.

TODO (domain author): the validity range of the flat per-op cost (bit-width regimes, the point at which carry-tree depth makes $t_{\mathrm{op}}$ bit-width-dependent), and any conditions under which modular wrap is a modelling error rather than the intended hardware behaviour.

## Validation

TODO (domain author): evidence that the modular-wrap function and the PPA accounting match the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the adder-tree architecture and the PPA basis.

---

- **Internals**: [accumulator internals](../../../internals/primitive/digital/accumulator.md)
- **Validation**: TODO - validation artefact not yet written
- **Configuration**: `neurox/primitive/digital/accumulator.py` (`AccumulatorConfig`)
