# Current ADC family

Every concrete current ADC digitizes a single-ended magnitude current into an unsigned integer code under one shared set of conventions.

## Shared conventions

A current ADC digitizes a single-ended, non-negative magnitude current $I_{\mathrm{in}}$ [uA] into one **unsigned** integer code. The converter commits to the current domain: the input, every reference level, and every noise term are expressed in microamperes.

A **1-D** ascending ladder $I_{\mathrm{ref}}$ of shape $[2^{b}-1]$ is a runtime input. The resolution $b$ and the selected ladder fully determine one conversion; an operating-mode identity is not part of the transfer relation.

The resolution $b$ sets the number of code levels; the maximum resolution $b_{\max}$ is fixed for a given ADC.

## Governing laws

The family quantization is a **monotone** mapping of the magnitude input against the selected reference row $\{I_{\mathrm{ref},c}\}$, all carrying microampere units. A conversion returns a **raw** unsigned code in

$$\mathrm{code} \in [0,\ 2^{b}-1],$$

the count of reference levels the input exceeds. Because the input is a non-negative magnitude, there is no zero-code shift: the raw unsigned code is the direct output.

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

Stated assumption: the input is a genuinely single-ended non-negative magnitude, so the unsigned-code convention is well-defined.

TODO (domain author): the value-range and operating-envelope limits across which the unsigned-code contract holds.

## References

TODO.

---

- **Internals**: [current ADC base internals](../../../../internals/primitive/analog/current_adc/base.md)
- **Configuration**: `SingleEndedCurrentAdcConfig` (see `api`)
