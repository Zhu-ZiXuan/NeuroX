# Ideal macro

## Summary

`IdealXbarMacro` is the lossless twin of the macro family: it carries no crossbar tile and no slicing, storing the integer weight matrix and computing an exact integer matrix multiply. It is the value-domain truth against which any physical mode is compared, the exact no-noise baseline. This document specifies the exact model it realizes.

## Physical model

N/A — the ideal macro models no hardware. It exercises the same value-domain contract as the physical modes (the accepted weight and input value ranges, the integer matmul protocol) but skips every tiling, slicing, aggregation, and analog step. It carries no ADC and computes the exact integer matmul, whereas a physical mode's read always carries per-tile ADC quantization; the ideal is therefore the exact integer reference the physical modes' results approach as that ADC quantization becomes negligible. Structurally it corresponds to the tile ADC surface's $b = 0$ sentinel documented in the [xbar/base §Ideal twin](../../xbar/family.md#ideal-twin).

## Governing equations

The ideal macro computes the exact integer dot product over the accepted value domain,

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

with the contraction carried out at full integer width. No quantization, no rescale, no decomposition: the result is the exact $\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}$. This exactness holds by construction, since the ideal macro carries no ADC.

## Noise & non-idealities

N/A — the ideal macro is lossless by construction. It publishes the sentinel ADC surface (`adc_mode_num = 1`, `adc_max_bits = 0`, $s = 1$), where `adc_max_bits = 0` is the "no output quantization" point that returns the integer dot product unmodified. This is the $b = 0$ sentinel of the tile ADC surface, owned by [xbar/base §Ideal twin](../../xbar/family.md#ideal-twin).

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `w_value_range` | inclusive integer weight range accepted | — | Design |
| `x_value_range` | inclusive integer input value range accepted | — | Design |

The ideal macro has no other parameters: it does no tiling or slicing, so no tile geometry or slice counts enter. Provenance terms: [module_parameter](../../../conventions/module_parameter.md).

## Symbols

The ideal macro introduces no new symbols. The logical dims ($N$, $K$, $M$), the value-domain ranges, and the ADC surface are in [macro/base](../family.md#symbols).

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
