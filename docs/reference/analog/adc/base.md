# ADC Abstract Layer

## Summary / role

Every concrete ADC in the family converts a differential analog input into a signed integer code under one shared contract: the same signed-code output convention, the same monotone code-edge floor semantics, and the same per-call operating point. This document specifies that contract; each topology's transfer characteristic and energy model is in its own document (see the [ADC family](README.md) index). The ADC sits at the end of the readout chain, downstream of the BL clamp and the column transport.

## Conversion contract

An ADC converts a differential input - a positive leg $x^{+}$ against a negative leg $x^{-}$ - into one signed integer code per call. Each ADC instance commits to a single per-instance input unit (volts for a voltage-domain topology, microamperes for a current-domain topology); both legs, all noise terms and the code boundaries are expressed in that one unit. The operating point is a per-call selection $(\mathrm{mode}, b)$: the mode picks a topology-specific configuration (e.g. one of the injected reference taps) and $b$ is the resolution in bits. An ADC exposing a single operating point accepts only $\mathrm{mode} = 0$ and $b = b_{\max}$; a multi-mode ADC accepts any pair inside its envelope. The ADC does not self-hold its reference: every call receives all reference taps as a `Tensor` $\{V_{\mathrm{ref},m}\}$, and $\mathrm{mode}$ indexes that tensor's trailing axis — the supported $\mathrm{mode}$ count is the tap count of the injected tensor (the owning xbar's reference-source `num_refs`), not a field of the ADC. The family exposes the maximum bit width as part of the contract.

## Signed-code output convention

A conversion returns a **signed** integer code. The family fixes the range generally through the bucket count $n_{\mathrm{codes}}$ and the zero code $z$: the raw unsigned bucket index lies in $[0,\ n_{\mathrm{codes}}-1]$ and subtracting $z$ shifts it to the signed range $[-z,\ n_{\mathrm{codes}}-1-z]$. This is a family-level contract: every concrete ADC, whatever its native internal representation, maps its result onto this signed range. For the linear / uniform-quantization members (see [floor semantics](#floor-semantics)) $n_{\mathrm{codes}} = 2^{b}$ and a symmetric $z$ yields the range $[-2^{\,b-1},\ 2^{\,b-1}-1]$.

The signed convention is required by the consumer rescale model $M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s$ used at the tile boundary (see the xbar [base](../../xbar/base.md#output-rescale)): a strictly positive rescale factor $s$ mapping a signed code to a signed dot product is well-defined only when the code carries the sign of the analog input directly. The tile's BL ADC is differential and its input is genuinely two-sided, so the signed code aligns the physical ADC with the integer-exact ideal twin and with the positivity invariant the calibration places on $s$.

A concrete ADC produces a raw unsigned bucket index, clamps it to its legal bucket range $[0,\ n_{\mathrm{codes}}-1]$, then subtracts a topology-specific **zero code** $z$ to shift to the signed range. The zero code is part of each topology's knowledge - a symmetric design places it at the bucket midpoint, but an asymmetric or single-ended design could place it elsewhere - so the base makes no commitment to a single shared zero point. The unsigned clamp must precede the zero shift: physical noise can push the raw index outside the legal range, which would otherwise yield an out-of-range signed code.

## Floor semantics

The family quantization is a **monotone** quantization of the input against an ordered set of code boundaries $\{B_c\}$, all carrying the instance's per-instance input unit. The boundaries need only be sorted; they may be non-uniformly spaced, and their count need not be a power of two. The conversion floors the input against them: the deterministic floor-bucketize law is

$$\mathrm{code} = \operatorname{clamp}\!\big(\operatorname{bucketize}(x,\ \{B_c\}),\ 0,\ n_{\mathrm{codes}}-1\big) - z,$$

where $\operatorname{bucketize}(x,\ \{B_c\})$ returns the index of the bucket into which the signal $x$ falls (the count of boundaries it exceeds), $n_{\mathrm{codes}}$ is the number of code buckets and $z$ is the topology zero code. This deterministic floor against ordered boundaries is the ADC physical quantization law; each member supplies its own boundary set.

For linear / uniform-quantization ADCs (all current concrete members) the boundaries are evenly spaced: $B_c = c\cdot \mathrm{LSB}$, with $\mathrm{LSB} = \mathrm{FSR} / 2^{b}$ the full-scale-range divided by the code count and $n_{\mathrm{codes}} = 2^{b}$. These are the linear / uniform sub-case, not universal: a non-uniform member overrides them with a calibrated boundary set.

## What the ADC does not own

- The clamp voltage and the current-to-voltage conversion - those are the [tia](../tia/README.md) (BL clamp) and [voltage_driver](../voltage_driver.md).
- The analog-domain rescale factor that maps codes back to the ideal-integer scale - that is the tile-boundary rescale in the xbar [base](../../xbar/base.md#output-rescale).
- Column multiplexing - that is the [voltage_mux](../voltage_mux.md).
- The reference taps themselves - the ADC does not source or store them. The owning xbar holds a [voltage_reference](../voltage_reference.md), samples it once per read, and injects all taps into `convert`; the ADC only selects one by $\mathrm{mode}$.

## Numerical method

N/A at the family level - each concrete topology specifies its own conversion (a single floor-bucketize, a successive-approximation loop, etc.).

## Noise & non-idealities

ADC quantization is intrinsic to every member. All further non-idealities (sampling, comparator offset and thermal noise, cap mismatch) are topology-specific and specified per concrete document, each gated by that topology's policy switch.

## Parameters

The abstract layer fixes no physical parameter; it carries only the operating-point contract (maximum bits, plus the per-call reference taps injected as a `Tensor`) and the static-PPA fields (area, leakage) inherited by every member. The reference taps are not an ADC parameter — they are owned and sized by the xbar's reference source. Per-topology parameters are tabulated in the concrete documents. Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the differential input is genuinely two-sided, so the signed-code convention is well-defined and the rescale factor is positive.

The operating-point envelope requires $b \geq 2$. The signed (two's-complement) code needs at least two bits: at $b = 1$ the maximum positive code is $2^{0} - 1 = 0$, so there is no positive code, and the ideal-twin rescale $s = M_{\max} / (2^{\,b-1} - 1)$ divides by zero. $b = 1$ is unsupported.

TODO (domain author): the value-range and operating-envelope limits across which the signed-code / floor contract holds.

## Validation

TODO - link validation evidence once written: physical-vs-ideal code agreement under a noise-off policy.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | positive / negative differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $\{V_{\mathrm{ref},m}\}$ | injected reference taps (per call, shape `(*inst, num_refs)`); $\mathrm{mode}$ selects one | V | `v_refs__V` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $\mathrm{FSR}$ | full-scale input range (per-instance input unit) | V or uA | derived |
| $\mathrm{LSB}$ | code least-significant-bit step, $\mathrm{FSR}/2^{b}$ (per-instance input unit) | V or uA | derived |
| $B_c$ | code boundary at index $c$ (per-instance input unit) | V or uA | derived |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived |
| $z$ | topology zero code | — | derived |
| $M_{\mathrm{ideal}}$ | ideal integer dot product (consumer scale) | — | — |
| $M_{\max}$ | maximum ideal integer dot product (rescale denominator) | — | — |
| $s$ | output rescale factor | — | `rescale_factor` |

---

- **Internals**: [adc base internals](../../../internals/analog/adc/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `ADCConfig`, `AdcOperationPoint`, `AdcCalibrationRecord` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
