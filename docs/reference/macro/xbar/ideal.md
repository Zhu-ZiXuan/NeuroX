# Ideal macro

## Summary

`IdealXbarMacro` is the lossless twin of the macro family: it carries no crossbar tile and no slicing, storing the integer weight matrix and computing an exact integer matrix multiply. It is the value-domain truth against which any physical mode is compared, and the no-noise baseline for the operator pipeline. This document specifies the exact model it realizes; its implementation is in [internals/macro/xbar/ideal](../../../internals/macro/xbar/ideal.md).

## Physical model

N/A — the ideal macro models no hardware. It exercises the same value-domain contract as the physical modes (the accepted weight and input value ranges, the integer matmul protocol) but skips every tiling, slicing, aggregation, and analog step. It is the integer reference any physical mode converges to in the no-noise limit — the limit in which every constituent tile read bypasses its ADC quantization (the $b = 0$ sentinel of the [xbar/base §Ideal twin](../../xbar/base.md#ideal-twin)).

## Governing equations

The ideal macro computes the exact integer dot product over the accepted value domain,

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

with the contraction carried out at full integer width. No quantization, no rescale, no decomposition: the result is the exact $\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}$. This exactness is the same as a physical mode whose every constituent tile ADC quantization is bypassed; in a physical mode the realized result instead carries per-tile ADC quantization, so $\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}$ holds only in the no-noise (ADC-bypassed) limit.

## Symbols

The ideal macro introduces no new symbols. The logical dims ($N$, $K$, $M$), the value-domain ranges, and the ADC surface are in [macro/base](../base.md#symbols).

## Noise & non-idealities

N/A — the ideal macro is lossless by construction. Its ADC surface is the sentinel (`adc_mode_num = 1`, `adc_max_bits = 0`, $s = 1$); the `adc_max_bits = 0` sentinel signals "no output quantization" to the operator. This is the $b = 0$ sentinel owned by the [xbar/base §Ideal twin](../../xbar/base.md#ideal-twin) — $b = 0$ returns the ideal integer dot product unmodified, $b = 1$ is degenerate and unsupported, and $b \geq 2$ applies the floor-quantized rescale.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `w_value_range` | inclusive integer weight range accepted | — | Design |
| `x_value_range` | inclusive integer input value range accepted | — | Design |

The ideal macro carries no tile config, no slice counts, and no reducer configs. Provenance terms: [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

- The macro trusts the upper layer for value ranges; it performs the exact matmul regardless of whether the inputs fit `w_value_range` / `x_value_range` (the ranges are published, not enforced).
- It is a value-domain reference only — it has no PPA contribution and no fabrication variation, so it cannot stand in for a physical mode in an energy or area study.
- It is a flow-bring-up / reference baseline, not a production result: the lossless value-domain upper bound and a way to isolate quantization issues from analog modelling, never a hardware-faithful accuracy or PPA number.

## Validation

N/A — the ideal macro is itself the validation reference for the physical modes.

## References

N/A.

---

- **Internals**: [ideal internals](../../../internals/macro/xbar/ideal.md)
- **Validation**: N/A — this document is the reference baseline
- **Configuration**: [config reference](../../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
