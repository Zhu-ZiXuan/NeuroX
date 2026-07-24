# Adder

Element-wise integer add of two broadcastable operands $a$ and $b$: an exact digital block whose only modeled physical content is the per-op PPA cost.

## Physical model

The integer sum is full-precision: no fixed-width register semantics apply, so the nominal bit width does not bound the result. The model is purely behavioural — no carry-chain timing, no per-bit gate model — so the only physical quantities it exposes are the PPA cost terms.

## Governing equations

The output is the element-wise sum of the two operands, broadcast to a common shape,

$$y = a + b,$$

over the full-precision integers, with no modular wrap and no clamp.

## Numerical method

N/A - exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A - exact digital function; no register wrap, no static mismatch, no per-call randomness.

## PPA cost model

The add is element-wise, so each output element is one adder evaluation. Latency is set by the busiest instance: the serial-op count of a call is the number of output elements on the instance carrying the most work,

$$n_{\mathrm{serial}} = \left\lceil \frac{\operatorname{numel}(y)}{\max(N_{\mathrm{inst}}, 1)} \right\rceil,$$

so its latency is $t = t_{\mathrm{op}}\, n_{\mathrm{serial}}$. Dynamic energy is total work, independent of how the outputs distribute across instances,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(y).$$

An empty call, $\operatorname{numel}(y) = 0$, costs zero latency and zero energy. Static area and leakage are the per-instance terms scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ (bit-width scaling, technology node); the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `bit_width` | nominal output bit width; does not bound the result (no wrap) | — | $\geq 1$ | Design |
| `energy_per_op__fJ` | dynamic energy per output element | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | latency per output element | ns | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | $\geq 0$ | Design |

Provenance terms: [module_parameter](../../../conventions/module_parameter.md). File-level schema: `neurox/primitive/digital/adder.py` (`AdderConfig`).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $a$ | left operand (runtime input) | — | `a` |
| $b$ | right operand, broadcastable to $a$ (runtime input) | — | `b` |
| $y$ | element-wise sum | — | return of `add` |
| $E_{\mathrm{op}}$ | dynamic energy per output element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | latency per output element | ns | `latency_per_op__ns` |
| $n_{\mathrm{serial}}$ | serial-op count of a call | — | `serial_op_count` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- No saturation and no wrap: the result is the full-precision integer sum, not range-bounded by the nominal bit width.
- The cost model is behavioural and per-op flat: energy and latency scale only with the output-element count, not with operand magnitude or carry depth.

TODO (domain author): the validity range of the flat per-op cost (bit-width regimes where carry depth makes $t_{\mathrm{op}}$ bit-width-dependent).

## Validation

TODO (domain author): evidence that the PPA accounting matches the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the adder architecture and the PPA basis.

---

- **Internals**: [adder internals](../../../internals/primitive/digital/adder.md)
- **Validation**: TODO - validation artefact not yet written
- **Configuration**: `neurox/primitive/digital/adder.py` (`AdderConfig`)
