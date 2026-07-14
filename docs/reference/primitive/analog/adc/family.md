# ADC family

Every concrete ADC in the family digitizes a differential analog input into a signed integer code under one shared set of conventions.

## Shared conventions

An ADC digitizes a differential input - a positive leg $x^{+}$ against a negative leg $x^{-}$ - into one signed integer code. Each ADC operates in a single physical domain and commits to one input unit (volts for a voltage-domain topology, microamperes for a current-domain topology); both legs, every noise term, and the code boundaries are expressed in that one unit.

The converter digitizes against an externally supplied reference rather than an internal constant, so the code edges follow that reference: the reference level sets the full-scale range, and the resolution $b$ (bits) sets the number of code levels within it. The maximum resolution $b_{\max}$ is a fixed characteristic of each ADC.

## Governing laws

The family quantization is a **monotone** quantization of the differential input against an ordered set of code boundaries $\{B_c\}$, all carrying the instance's input unit. The boundaries need only be sorted; they may be non-uniformly spaced, and their count need not be a power of two. The conversion floors the input against them and shifts the result to a signed range by a topology zero code $z$:

$$\mathrm{code} = \operatorname{clamp}\!\big(\operatorname{bucketize}(x,\ \{B_c\}),\ 0,\ n_{\mathrm{codes}}-1\big) - z,$$

where $\operatorname{bucketize}(x,\ \{B_c\})$ returns the index of the bucket into which the signal $x$ falls (the count of boundaries it exceeds), $n_{\mathrm{codes}}$ is the number of code buckets, and $z$ is the topology zero code. Physical noise can push the raw bucket index outside $[0,\ n_{\mathrm{codes}}-1]$, so it is clamped to that legal range before the zero shift. This deterministic floor against ordered boundaries is the ADC physical quantization law; each member supplies its own boundary set.

### Signed-code range

A conversion returns a **signed** integer code. The raw unsigned bucket index lies in $[0,\ n_{\mathrm{codes}}-1]$, and subtracting the zero code $z$ shifts it to the signed range $[-z,\ n_{\mathrm{codes}}-1-z]$. The zero code is topology-specific: a symmetric design places it at the bucket midpoint, but an asymmetric or single-ended design places it elsewhere.

### Uniform sub-case

For linear / uniform-quantization ADCs (all current concrete members) the boundaries are evenly spaced, $B_c = c\cdot \mathrm{LSB}$, with $\mathrm{LSB} = \mathrm{FSR} / 2^{b}$ the full-scale range divided by the code count and $n_{\mathrm{codes}} = 2^{b}$. A symmetric zero code $z = 2^{\,b-1}$ then yields the range $[-2^{\,b-1},\ 2^{\,b-1}-1]$. A non-uniform member overrides it with a calibrated boundary set.

## Noise & non-idealities

ADC quantization is intrinsic to every member; all further non-idealities are topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | positive / negative differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $\mathrm{FSR}$ | full-scale input range (per-instance input unit) | V or uA | derived |
| $\mathrm{LSB}$ | code least-significant-bit step, $\mathrm{FSR}/2^{b}$ (per-instance input unit) | V or uA | derived |
| $B_c$ | code boundary at index $c$ (per-instance input unit) | V or uA | derived |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived |
| $z$ | topology zero code | — | derived |

## Assumptions, scope & validity

Stated assumption: the differential input is genuinely two-sided, so the signed-code convention is well-defined.

The signed-code convention is valid for $b \geq 1$. At the boundary $b = 1$ the code range degenerates: $n_{\mathrm{codes}} = 2^{1} = 2$ with symmetric zero code $z = 2^{\,0} = 1$ gives the signed range $[-1,\ 0]$ — one sign bit carrying one negative code and zero, with no positive code — but the convention stays well-defined, so $b = 1$ is supported.

TODO (domain author): the value-range and operating-envelope limits across which the signed-code / floor contract holds.

## References

TODO.

---

- **Internals**: [adc base internals](../../../../internals/primitive/analog/adc/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `ADCConfig`, `AdcOperationPoint`, `AdcCalibrationRecord` (see `api`)
