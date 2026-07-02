# Subtractor

## Summary

The subtractor is the element-wise integer subtract primitive of the digital periphery: it computes the difference of two broadcastable integer tensors with no saturation and no wrap. It is the sign twin of the [adder](adder.md), sharing its structure and cost model. It is an exact digital block; the only modelled physical content is its behavioural PPA cost.

## Physical model

The block abstracts a parallel array of integer subtractors, one per output element. It applies no fixed-width register semantics: the nominal `bit_width` is informational and does not bound the result. The model is purely behavioural — no borrow-chain timing, no per-bit gate model — so the only physical quantities it exposes are the PPA cost terms.

## Governing equations

The output is the element-wise difference of the two operands, broadcast to a common shape,

$$y = a - b,$$

over the full-precision integers, with no modular wrap and no clamp. Here $a$ is the minuend and $b$ the subtrahend.

## Numerical method

N/A - exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A - exact digital function; no register wrap, no static mismatch, no per-call randomness.

## PPA cost model

The subtract is element-wise, so each output element is one subtractor evaluation. Latency is set by the busiest instance: the serial-op count of a call is the number of output elements on the instance carrying the most work,

$$n_{\mathrm{serial}} = \left\lceil \frac{\operatorname{numel}(y)}{\max(N_{\mathrm{inst}}, 1)} \right\rceil,$$

so its latency is $t = t_{\mathrm{op}}\, n_{\mathrm{serial}}$. Dynamic energy is total work, independent of how the outputs distribute across instances,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(y).$$

An empty call, $\operatorname{numel}(y) = 0$, costs zero latency and zero energy. Static area and leakage are the inherited per-instance terms scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ (bit-width scaling, technology node); the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `bit_width` | nominal output bit width (informational; no wrap applied) | — | Design |
| `energy_per_op__fJ` | dynamic energy per output element | fJ | Design |
| `latency_per_op__ns` | latency per output element | ns | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | Design |

Provenance terms: [module_parameter](../../conventions/module_parameter.md). File-level schema: `neurox/digital/subtractor.py` (`SubtractorConfig`).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $a$ | minuend (runtime input) | — | `a` |
| $b$ | subtrahend, broadcastable to $a$ (runtime input) | — | `b` |
| $y$ | element-wise difference | — | return of `operate` |
| $w$ | nominal output bit width (informational) | — | `bit_width` |
| $E_{\mathrm{op}}$ | dynamic energy per output element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | latency per output element | ns | `latency_per_op__ns` |
| $n_{\mathrm{serial}}$ | serial-op count of a call | — | `serial_op_count` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- No saturation and no wrap: `bit_width` is informational and the result is not range-bounded, so the consumer must guarantee the operands fit.
- The cost model is behavioural and per-op flat: energy and latency scale only with the output-element count, not with operand magnitude or borrow depth.

TODO (domain author): the validity range of the flat per-op cost (bit-width regimes where borrow depth makes $t_{\mathrm{op}}$ bit-width-dependent).

## Validation

TODO (domain author): evidence that the PPA accounting matches the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the subtractor architecture and the PPA basis.

---

- **Internals**: [subtractor internals](../../internals/digital/subtractor.md)
- **Validation**: TODO - validation artefact not yet written
- **Configuration**: `neurox/digital/subtractor.py` (`SubtractorConfig`)
- **Decisions**: N/A — no ADR governs this module.
