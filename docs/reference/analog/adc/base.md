# ADC Abstract Layer

## Summary / role

Every concrete ADC in the family - the boundary-bucketize [general](general.md), the [mcs_sar](mcs_sar.md) and [sar_mono](sar_mono.md) SAR variants - converts a differential analog input into a signed integer code under one shared contract: the same signed-code output convention, the same code-edge floor semantics, and the same per-call multi-mode operating point. This document specifies that contract; each topology's transfer characteristic and energy model is in its own document. The ADC sits at the end of the readout chain, downstream of the BL clamp and the column transport.

## Conversion contract

An ADC converts a differential input - a positive leg $V^{+}$ against a negative leg $V^{-}$ - into one signed integer code per call. The operating point is a per-call selection $(\mathrm{mode}, b)$: the mode picks a topology-specific configuration (e.g. a reference-voltage entry) and $b$ is the resolution in bits. A single-mode ADC accepts only $\mathrm{mode} = 0$ and $b = b_{\max}$; a multi-mode ADC accepts any pair inside its configured envelope. The family exposes the count of supported operating points and the maximum bit width as part of the contract.

## Signed-code output convention

A conversion returns a **signed** integer code in $[-2^{\,b-1},\ 2^{\,b-1}-1]$. This is a family-level contract: every concrete ADC, whatever its native internal representation, maps its result onto this signed range.

The signed convention is required by the consumer rescale model $M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s$ used at the tile boundary (see the xbar [base](../../xbar/base.md#output-rescale)): a strictly positive rescale factor $s$ mapping a signed code to a signed dot product is well-defined only when the code carries the sign of the analog input directly. The tile's BL ADC is differential and its input is genuinely two-sided, so the signed code aligns the physical ADC with the integer-exact ideal twin and with the positivity invariant the calibration places on $s$.

A concrete ADC produces a raw unsigned bucket index, clamps it to its legal bucket range, then subtracts a topology-specific **zero code** to shift to the signed range. The zero code is part of each topology's knowledge - a symmetric design places it at the bucket midpoint, but an asymmetric or single-ended design could place it elsewhere - so the base makes no commitment to a single shared zero point. The unsigned clamp must precede the zero shift: stochastic-rounding jitter can push the raw index outside the legal range, which would otherwise yield an out-of-range signed code.

## Floor semantics

ADC code boundaries sit at code edges $B_c = c\cdot \mathrm{LSB}$ and the conversion floors the input against them. Stochastic rounding adds uniform jitter $\operatorname{uniform}(0, \mathrm{LSB})$ before the floor; the result is unbiased. Stochastic-versus-deterministic rounding follows the module's training/eval state, not a per-conversion override. This is the same floor-bucketize quantization specified for the [common quant kernels](../../../internals/common/quant.md).

## What the ADC does not own

- The clamp voltage and the current-to-voltage conversion - those are the [tia](../tia/README.md) (BL clamp) and [driver](../driver.md).
- The analog-domain rescale factor that maps codes back to the ideal-integer scale - that is the tile-boundary rescale in the xbar [base](../../xbar/base.md#output-rescale).
- Column multiplexing - that is the [analog_mux](../analog_mux.md).

## Numerical method

N/A at the family level - each concrete topology specifies its own conversion (a single floor-bucketize for [general](general.md), a successive-approximation loop for the SAR variants).

## Noise & non-idealities

ADC quantization is intrinsic to every member. All further non-idealities (sampling, comparator offset and thermal noise, cap mismatch) are topology-specific and specified per concrete document, each gated by that topology's policy switch.

## Parameters

The abstract layer fixes no physical parameter; it carries only the operating-point contract (mode count, maximum bits) and the static-PPA fields (area, leakage) inherited by every member. Per-topology parameters are tabulated in the concrete documents. Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the differential input is genuinely two-sided, so the signed-code convention is well-defined and the rescale factor is positive.

TODO (domain author): the value-range and operating-envelope limits across which the signed-code / floor contract holds.

## Validation

TODO - link validation evidence once written: physical-vs-ideal code agreement under a noise-off policy.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V^{+}, V^{-}$ | positive / negative differential input legs | V | `v_pos__V`, `v_neg__V` |
| $b$ | ADC resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | maximum supported resolution | — | `max_bits` |
| $\mathrm{LSB}$ | code least-significant-bit step | — | derived |
| $B_c$ | code boundary at index $c$ | — | derived |
| $M_{\mathrm{ideal}}$ | ideal integer dot product (consumer scale) | — | — |
| $s$ | output rescale factor | — | `rescale_factor` |

---

- **Internals**: [adc base internals](../../../internals/analog/adc/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `ADCConfig`, `AdcOperationPoint`, `AdcCalibrationRecord` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
