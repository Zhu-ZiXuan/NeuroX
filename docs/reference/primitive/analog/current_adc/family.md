# Current ADC family

Every concrete current ADC digitizes a single-ended magnitude current into an unsigned integer code under one shared set of conventions.

## Shared conventions

A current ADC digitizes a single-ended, non-negative magnitude current $I_{\mathrm{in}}$ [uA] into one **unsigned** integer code. The input is a magnitude only — the sign is handled outside the ADC by the caller, which combines it with the magnitude code downstream. The converter commits to the current domain: the input, every reference level, and every noise term are expressed in microamperes.

The reference ladder is **not** owned by the ADC, and neither is the operating mode. A **1-D** ladder $I_{\mathrm{ref}}$ [`i_refs__uA`, shape $[2^{b}-1]$, taps ascending] arrives per call, already reduced to the mode's row by the caller's reference block — the single ladder source. Mode is invisible to the ADC. Because the ladder is a per-call input, a single-ended and a (deferred) differential current ADC are **parallel classes** reading the same injected taps, never one wrapping the other.

The resolution $b$ (bits) arrives per call as `bits` and sets the number of code levels; the maximum resolution $b_{\max}$ is a fixed characteristic of each ADC.

## Governing laws

The family quantization is a **monotone** mapping of the magnitude input against the selected reference row $\{I_{\mathrm{ref},c}\}$, all carrying microampere units. A conversion returns a **raw** unsigned code in

$$\mathrm{code} \in [0,\ 2^{b}-1],$$

the count of reference levels the input exceeds. Because the input is a non-negative magnitude, there is no zero-code shift: the raw unsigned code is the direct output, and the sign (offset binary, sign-magnitude, or none) is reattached by the caller.

## Noise & non-idealities

Quantization against the reference levels is intrinsic to every member; all further non-idealities are topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $b$ | ADC resolution (bits), per call | — | `bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $I_{\mathrm{ref}}$ | per-call 1-D reference ladder, $[2^{b}-1]$ | uA | `i_refs__uA` |

## Assumptions, scope & validity

Stated assumption: the input is a genuinely single-ended non-negative magnitude, so the unsigned-code convention is well-defined and the sign lives with the caller.

TODO (domain author): the value-range and operating-envelope limits across which the unsigned-code contract holds.

## References

TODO.

---

- **Internals**: [current ADC base internals](../../../../internals/primitive/analog/current_adc/base.md)
- **Configuration**: `SingleEndedCurrentAdcConfig`; the shared `AdcCalibrationRecord` from `adc_common` is a macro/unit-layer rescale record (see `api`)
