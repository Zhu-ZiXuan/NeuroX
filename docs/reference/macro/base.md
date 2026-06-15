# Macro Abstract Contract

## Summary

A macro is the architectural unit that turns one or more [crossbar tiles](../xbar/README.md) into a complete quantised-integer matrix multiply. It models the simulated hardware one level above the tile: the tile carries a single primitive analog VMM over its native integer value domain, and the macro is what decomposes a high-precision logical weight and activation onto that domain, schedules the per-tile reads, and recombines the partial results into one integer dot product. This document specifies the macro contract that every family member satisfies; the xbar-tile family that realizes it is in [xbar/](xbar/README.md).

## Physical model

The macro models no silicon of its own. It owns no cells, no interconnect, no readout — those belong to the tiles and the digital reduction blocks it composes. What the macro adds is the *architecture*: the rule by which a logical weight matrix that exceeds one tile's value range or row/column count is spread across tiles, and the dual rule by which the per-tile results are summed back. Physically this corresponds to a chip region holding an array of crossbar tiles plus the digital adders that fold their outputs; the macro is the abstraction of that region's organization.

The macro presents a single integer matrix multiply to its consumer. Internally the multiply is realized as the sum of per-tile primitive VMMs over a decomposed value domain, but the contract is value-domain only: it states the integer ranges accepted and the integer (pre-requantize) result produced, not the physics of any constituent read.

## Governing equations

The macro computes an integer dot product matching exact matrix multiplication over its accepted value domain. For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$, both restricted to the macro's integer value ranges, the result is

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

returned as an integer pre-requantize tensor. Bias addition and the requantization back to the activation grid are not part of the macro; they belong to the operator that wraps it.

The decomposition that realizes this dot product on physical tiles is value-domain exact: a logical weight is sliced into per-weight slices of count `Sw`, each slice an integer in a tile-carriable range; a logical activation is sliced into per-cycle slices of count `Sa`. Both decompositions are positional, so the partial products recombine by a radix-weighted shift-add. The tiling and aggregation scheme that arranges these slices on the physical tile layout is specified in [xbar/base](../xbar/base.md); a degenerate member with `Sw = Sa = 1` performs no slicing.

## Noise & non-idealities

N/A at the contract level — the macro is a value-domain organizer and adds no analog non-ideality of its own. Every non-ideality enters through the tiles it reads (IR drop, device noise, finite-gain clamps, ADC quantization) and through the digital reduction blocks (exact by construction). The tile-level sources are specified in [xbar/base](../xbar/base.md) and the topology families beneath it.

## Parameters

The abstract contract has no parameters of its own; a concrete member's parameters are its tile configuration plus its slice counts. The architectural value-domain capabilities a macro publishes are below; per-mode parameter tables are in the family documents.

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `w_value_range` | inclusive integer logical-weight range accepted | — | Design |
| `x_value_range` | inclusive integer logical-activation range accepted | — | Design |
| `adc_mode_num` | number of supported ADC operating points | — | Design |
| `adc_max_bits` | maximum ADC resolution across operating points | — | Design |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md). Runtime inputs (the weight and activation tensors, the ADC operating point) are inputs, not parameters.

### ADC operating-point surface

The macro inherits a discrete ADC operating-point surface from the tiles it reads. It publishes the number of supported operating points `adc_mode_num` (valid `adc_mode` indices are $[0, \text{adc\_mode\_num})$) and the maximum resolution `adc_max_bits`, and for any operating point it publishes the recovery-side rescale factor $s$ relating the integer dot product to the digitized code,

$$M_{\text{ideal}} \approx \text{code}\cdot s.$$

The macro derives this surface from its tiles; the rescale-factor convention and its calibration are the tile-level contract in [xbar/base](../xbar/base.md#output-rescale). A degenerate member that performs an exact integer matmul publishes a sentinel surface (`adc_mode_num = 1`, `adc_max_bits = 0`, $s = 1$), where `adc_max_bits = 0` signals "no output quantization".

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathbf{W}$ | logical weight matrix (runtime input) | — | `weight` |
| $\mathbf{X}$ | logical activation matrix (runtime input) | — | `input` |
| $\mathbf{Y}$ | pre-requantize integer output | — | `matmul` return |
| $N, K, M$ | output, contraction, and activation-row dims | — | `w_logical_shape`, input shape |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $M_{\text{ideal}}$ | ideal integer dot product | — | — |

Slice counts (`Sw`, `Sa`) and tile-layout counts (`Tc`, `Tr`) are structure counts referred to by code field; they enter no equation here and are specified in [xbar/base](../xbar/base.md). The integer value ranges are published as `w_value_range` and `x_value_range`.

## Assumptions, scope & validity

Stated assumptions of the macro contract:

- The macro returns a pre-requantize integer result; bias and requantization are the consuming operator's responsibility, not the macro's.
- Value ranges are a published capability, not an enforced bound: a member trusts the upper mapping layer to supply integer weights and activations already inside `w_value_range` and `x_value_range`. Out-of-range inputs are not checked and produce undefined results.
- The decomposition is value-domain exact: the only deviation from the exact integer dot product is the analog non-ideality of the constituent tile reads, not the slicing or aggregation arithmetic.

TODO (domain author): state the validity boundary of the slice-and-shift-add decomposition (saturation of the positional recombination, the largest dot-product magnitude representable before the ADC code clamps) and any regime where the value-domain-exact assumption breaks.

## Validation

TODO: link [validation/macro](../../validation/README.md) — agreement of each mode against the ideal twin under a noise-off policy, and bit-exactness of the organize/aggregate dual.

## References

TODO: cite the bit-sliced compute-in-memory architecture and the positional shift-add recombination.

---

- **Internals**: [macro base internals](../../internals/macro/base.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
