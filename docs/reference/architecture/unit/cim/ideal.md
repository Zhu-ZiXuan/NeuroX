# Ideal unit

The ideal unit is the lossless limit of the unit family: it carries the integer weight matrix and returns the exact integer matrix product over the accepted value domain, with no tile, slicing, or ADC — the value-domain reference for the family's physical modes.

## Physical model

The model realizes no hardware. It deliberately removes every tiling, slicing, aggregation, and analog step, so no ADC quantization enters and the integer matrix product is returned exactly.

## Governing equations

The model computes the exact integer dot product over the accepted value domain,

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

with the contraction carried out at full integer width. No quantization, no rescale, no decomposition: the result is the exact $\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}$.

## Noise & non-idealities

N/A — the model is lossless by construction and adds no non-ideality. Its ADC surface is the sentinel point (`adc_mode_num = 1`, `adc_max_bits = 0`, rescale $s = 1$), where `adc_max_bits = 0` is the no-output-quantization point that returns the integer dot product unmodified — the $b = 0$ sentinel of the tile ADC surface in [CimMacro base §Ideal twin](../../../primitive/macro/cim/README.md#ideal-twin).

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `w_value_range` | inclusive integer weight range accepted | — | — | Design |
| `x_value_range` | inclusive integer activation range accepted | — | — | Design |

No tile geometry or slice counts enter: the model does no tiling or slicing. Provenance terms: [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

This mode introduces no new symbols. The logical dims ($N$, $K$, $M$), the value-domain ranges, and the ADC surface are in [unit/base](../base.md#symbols).

## Assumptions, scope & validity

- The accepted value ranges are published, not enforced: the model computes the exact matmul whether or not the inputs fit `w_value_range` / `x_value_range`, carrying out-of-range values through exactly.
- It is a lossless value-domain reference only, with no PPA contribution and no fabrication variation: the lossless upper bound, not a hardware-faithful accuracy, energy, or area figure.

## Validation

N/A — this mode is itself the reference the physical modes are validated against.

## References

N/A.

---

- **Internals**: [ideal internals](../../../../internals/architecture/unit/cim/ideal.md)
- **Validation**: N/A — this document is the reference baseline
- **Configuration**: [config reference](../../../../api/README.md)
