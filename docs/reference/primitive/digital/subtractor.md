# Subtractor

The subtractor computes the element-wise difference of the minuend $a$ and subtrahend $b$, two broadcastable integer tensors, with no saturation and no wrap. It is an exact digital block whose only modelled physical content is its behavioural PPA cost.

## Physical model

The block abstracts a parallel array of integer subtractors. No fixed-width register semantics apply: the nominal bit width is informational and does not bound the result. The model is behavioural, so the only physical quantities it exposes are the PPA cost terms.

## Governing equations

The output is the element-wise difference of the two operands, broadcast to a common shape,

$$y = a - b,$$

over the full-precision integers, with no modular wrap and no clamp. Here $a$ is the minuend and $b$ the subtrahend.

## Numerical method

N/A — exact integer arithmetic; no iterative or approximate solve.

## Noise & non-idealities

N/A — exact digital function; no register wrap, no static mismatch, no per-call randomness.

## PPA cost model

The subtract is element-wise, so each output element is one subtractor evaluation. The operation runs at once across every fabricated instance, so the block owns no time axis and its duration is the flat combinational window $t = t_{\mathrm{op}}$; a caller that issues several rounds on it counts them itself. Dynamic energy is total work, independent of how the outputs distribute across instances,

$$E = E_{\mathrm{op}}\, \operatorname{numel}(y).$$

An empty call, $\operatorname{numel}(y) = 0$, costs zero energy. Static area and leakage are the per-instance terms scaled by the instance count.

TODO (domain author): the provenance and derivation of $E_{\mathrm{op}}$, $t_{\mathrm{op}}$, $A_{\mathrm{inst}}$, $P_{\mathrm{inst}}$ (bit-width scaling, technology node); the source docs give only the accounting form, not the values.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `bit_width` | nominal output bit width (informational; no wrap applied) | — | $\geq 1$ | Design |
| `energy_per_op__fJ` | dynamic energy per output element | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | combinational window of one subtract | ns | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` | static leakage per instance | uW | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $a$ | minuend (runtime input) | — | `a` |
| $b$ | subtrahend, broadcastable to $a$ (runtime input) | — | `b` |
| $y$ | element-wise difference | — | return of `subtract` |
| $E_{\mathrm{op}}$ | dynamic energy per output element | fJ | `energy_per_op__fJ` |
| $t_{\mathrm{op}}$ | combinational window of one subtract | ns | `latency_per_op__ns` |
| $N_{\mathrm{inst}}$ | fabricated instance count | — | `inst_count` |
| $A_{\mathrm{inst}}$ | area per instance | um^2 | `area_per_inst__um2` |
| $P_{\mathrm{inst}}$ | leakage per instance | uW | `leakage_per_inst__uW` |

## Assumptions, scope & validity

- No saturation or wrap is applied, so the exact-integer difference matches fixed-width hardware only where the operands stay within the nominal bit width.
- The cost model is behavioural and per-op flat: energy scales only with the output-element count and the duration not at all, neither varying with operand magnitude or borrow depth.

TODO (domain author): the validity range of the flat per-op cost (bit-width regimes where borrow depth makes $t_{\mathrm{op}}$ bit-width-dependent).

## Validation

TODO (domain author): evidence that the PPA accounting matches the intended hardware; no validation artefact exists yet.

## References

TODO (domain author): cite the subtractor architecture and the PPA basis.

---

- **Internals**: [subtractor internals](../../../internals/primitive/digital/subtractor.md)
- **Validation**: TODO — validation artefact not yet written
- **Configuration**: `neurox/primitive/digital/subtractor.py` (`SubtractorConfig`)
