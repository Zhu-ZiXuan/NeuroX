# Voltage ADC family

Every concrete ADC in the family digitizes a differential analog input into a raw unsigned integer code under one shared set of conventions. Sign and zero-point handling live entirely outside the ADC, on the consumer (the composing macro).

## Shared conventions

An ADC digitizes a differential input - a positive leg $x^{+}$ against a negative leg $x^{-}$ - into one raw unsigned integer code. Each ADC operates in a single physical domain and commits to one input unit (volts for a voltage-domain topology, microamperes for a current-domain topology); both legs, every noise term, and the code boundaries are expressed in that one unit.

The converter digitizes against an externally supplied reference rather than an internal constant, so the code edges follow that reference: the reference level sets the full-scale range, and the resolution $b$ (bits) sets the number of code levels within it. The maximum resolution $b_{\max}$ is a fixed characteristic of each ADC.

## Governing laws

The family quantization is a **monotone** quantization of the differential input against an ordered set of code boundaries $\{B_c\}$, all carrying the instance's input unit. The boundaries need only be sorted; they may be non-uniformly spaced, and their count need not be a power of two. The conversion floors the input against them and clamps the result into the legal raw code range:

$$\mathrm{code} = \operatorname{clamp}\!\big(\operatorname{bucketize}(x,\ \{B_c\}),\ 0,\ n_{\mathrm{codes}}-1\big),$$

where $\operatorname{bucketize}(x,\ \{B_c\})$ returns the index of the bucket into which the signal $x$ falls (the count of boundaries it exceeds), and $n_{\mathrm{codes}}$ is the number of code buckets. Physical noise can push the raw bucket index outside $[0,\ n_{\mathrm{codes}}-1]$, so it is clamped to that legal range. The ADC returns this raw unsigned code directly; it never subtracts a zero point. This deterministic floor against ordered boundaries is the ADC physical quantization law; each member supplies its own boundary set.

### Raw-code range and consumer-side recovery

A conversion returns a **raw unsigned** integer code (offset-binary index) in $[0,\ n_{\mathrm{codes}}-1]$. The ADC does not fold sign or offset into that code. Recovery to a signed physical magnitude is the consumer's job and is affine: subtract the zero-point offset $z$, then scale,

$$M_{\mathrm{ideal}} \approx (\mathrm{code} - z)\cdot \mathrm{rescale\_factor},\qquad \mathrm{rescale\_factor} > 0.$$

The zero-point offset $z$ is a property of the ADC (exposed by the member's zero-point accessor), but subtracting it — and any downstream sign assembly — happens in the consuming macro, not inside the ADC. The offset is topology-specific: a symmetric design places it at the bucket midpoint, but an asymmetric or single-ended design places it elsewhere (a genuinely single-ended magnitude ADC uses $z = 0$).

### Uniform sub-case

For linear / uniform-quantization ADCs the boundaries are evenly spaced, $B_c = c\cdot \mathrm{LSB}$, with $\mathrm{LSB} = \mathrm{FSR} / 2^{b}$ the full-scale range divided by the code count and $n_{\mathrm{codes}} = 2^{b}$. The raw code range is then $[0,\ 2^{b}-1]$ and the symmetric zero-point offset is $z = 2^{\,b-1}$, so the consumer's recovered signed range is $[-2^{\,b-1},\ 2^{\,b-1}-1]$. A non-uniform member overrides the boundary set and its own zero-point offset.

## Noise & non-idealities

ADC quantization is intrinsic to every member; all further non-idealities are topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | positive / negative differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $b$ | ADC resolution (bits) | — | `bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $\mathrm{FSR}$ | full-scale input range (per-instance input unit) | V or uA | derived |
| $\mathrm{LSB}$ | code least-significant-bit step, $\mathrm{FSR}/2^{b}$ (per-instance input unit) | V or uA | derived |
| $B_c$ | code boundary at index $c$ (per-instance input unit) | V or uA | derived |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived |
| $z$ | zero-point offset (subtracted consumer-side) | — | `zero_offset(bits)` |

## Assumptions, scope & validity

Stated assumption: the differential input is genuinely two-sided, so the consumer's zero-point recovery is well-defined.

The raw-code convention is valid for $b \geq 1$. At the boundary $b = 1$ the range degenerates: $n_{\mathrm{codes}} = 2^{1} = 2$ gives raw codes $\{0, 1\}$ with symmetric zero-point offset $z = 2^{\,0} = 1$, so the consumer recovers the signed range $[-1,\ 0]$ — one sign bit carrying one negative code and zero, with no positive code — but the convention stays well-defined, so $b = 1$ is supported.

TODO (domain author): the value-range and operating-envelope limits across which the raw-code / floor contract holds.

## References

TODO.

---

- **Internals**: [adc base internals](../../../../internals/primitive/analog/voltage_adc/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `DifferentialVoltageAdcConfig`; `convert` takes a preselected `v_ref__V` reference tap plus `bits`; `AdcCalibrationRecord` (from `adc_common`) (see `api`)
