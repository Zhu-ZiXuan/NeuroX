# Current ADC family

Every concrete current ADC digitizes a single-ended magnitude current into an unsigned integer code under one shared set of conventions.

## Shared conventions

A current ADC digitizes a single-ended, non-negative magnitude current $I_{\mathrm{in}}$ [uA] into one **unsigned** integer code. The input is a magnitude only — the sign is handled outside the ADC by the caller, which combines it with the magnitude code downstream. The converter commits to the current domain: the input, every reference level, and every noise term are expressed in microamperes.

The resolution $b$ (bits) sets the number of code levels, and the maximum resolution $b_{\max}$ is a fixed characteristic of each ADC.

## Governing laws

The family quantization is a **monotone** mapping of the magnitude input against an ordered set of reference levels $\{I_{\mathrm{ref},c}\}$, all carrying microampere units. A conversion returns an unsigned code in

$$\mathrm{code} \in [0,\ 2^{b}-1],$$

the count of reference levels the input exceeds. Because the input is a non-negative magnitude, there is no zero-code shift: the unsigned code is the direct output, and the sign is reattached by the caller.

## Noise & non-idealities

Quantization against the reference levels is intrinsic to every member; all further non-idealities are topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $I_{\mathrm{ref},c}$ | reference level at index $c$ | uA | `ref_levels__uA` |

## Assumptions, scope & validity

Stated assumption: the input is a genuinely single-ended non-negative magnitude, so the unsigned-code convention is well-defined and the sign lives with the caller.

TODO (domain author): the value-range and operating-envelope limits across which the unsigned-code contract holds.

## References

TODO.

---

- **Internals**: [current ADC base internals](../../../../internals/primitive/analog/current_adc/base.md)
- **Configuration**: `CurrentAdcConfig`; the shared `AdcOperationPoint`, `AdcCalibrationRecord` from `adc_common` (see `api`)
