# General ADC

## Summary / role

`GeneralADC` is the boundary-bucketize behavioural ADC: a sorted list of comparator thresholds plus optional Gaussian noise stages. It is the simplest member of the [ADC family](README.md) - a single-mode digitizer whose resolution is fixed by its boundary list. It honours the family signed-code and floor contract in [base](base.md).

Note: `GeneralADC` is a historical / placeholder ADC. Its behavioural-comparator model is not yet fully designed; the boundary list, noise stages and input-unit handling are provisional and may change.

## Physical model

The ADC is modelled as a bank of comparators against a sorted threshold list `boundaries` (in input units - uA for current-mode, V for voltage-mode). The differential input is compared against every threshold and the index of the bucket it falls into is the raw code. The model exposes three optional additive Gaussian noise stages - input-referred sampling noise, a single input-referred comparator-noise term on the signal, and drive-thermal noise.

## Governing equations

The conversion maps the raw differential input $x^{+} - x^{-}$ to a signed code. All quantities below carry the single per-instance input unit (uA for current-mode, V for voltage-mode), so the bucketize arguments and the boundaries are dimensionally consistent. With optional sampling noise $n_s$ added input-referred and an optional single input-referred comparator-noise term $n_{c}$ on the signal,

$$\mathrm{code} = \operatorname{clamp}\!\Big(\operatorname{bucketize}\big(x^{+}-x^{-}+n_s+n_{c},\ \{B_c\}\big),\ 0,\ n_{\mathrm{codes}}-1\Big) - z,$$

where $\operatorname{bucketize}$ floors the noisy signal against the fixed boundaries $\{B_c\}$ - both signal and boundaries in the same input unit - $n_{\mathrm{codes}}$ is the number of code buckets implied by the boundary list, and $z$ is the topology zero code. The signed output lies in $[-z,\ n_{\mathrm{codes}}-1-z]$. This is the deterministic floor-bucketize law of the [family contract](base.md#floor-semantics); the clamp catches a bucket index pushed out of range by noise before the zero shift. The calibrated `boundaries` may be non-uniformly spaced and need not number a power of two, consistent with the generalized base contract.

The class is single-mode: $\mathrm{mode} = 0$ and $b$ equal to the boundary-implied bit width are the only legal pair.

## Numerical method

N/A - the conversion is a single floor-bucketize evaluation per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| sampling noise | input-referred sample jitter | additive zero-mean Gaussian on the input | `sampling_noise__V` | `sampling_noise` |
| comparator noise | comparator (thermal/decision) noise | single additive zero-mean Gaussian on the signal | `comparator_noise__V` | `comparator_noise` |
| drive thermal | thermal noise at drive time | additive zero-mean Gaussian | `drive_thermal__V` | `drive_thermal` |
| quantization | intrinsic floor-bucketize | deterministic floor against ordered boundaries | `boundaries` | — |

All sources are dynamic (sampled per convert); `GeneralADC` carries no static mismatch.

TODO (domain author): physical derivation and citation for each noise sigma.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `boundaries` | sorted comparator thresholds (input units) | uA or V | Calibrated (physical data) |
| `drive_value` | clamp reference voltage (ignored when a separate driver clamps) | V | Design |
| `sampling_noise__V` | input-referred sampling-noise sigma | V | Measured |
| `comparator_noise__V` | input-referred comparator-noise sigma | V | Measured |
| `drive_thermal__V` | drive-thermal noise sigma | V | Measured |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md). The comparator thresholds are calibrated against physical data via the ADC calibration procedure ([calibration guide](../../../guides/calibration/README.md)).

## Assumptions, scope & validity

Stated assumptions:

- The bit width is fixed by the boundary list (single-mode); only the boundary-implied resolution is legal.
- The `boundaries` are sorted, so the bucketize order is preserved.

TODO (domain author): the validity boundary of the behavioural-comparator model and the input-range limits implied by the boundary list.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $B_c$ | comparator threshold at index $c$ (per-instance input unit) | V or uA | `boundaries` |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived from `boundaries` |
| $z$ | topology zero code | — | `_zero_code` |
| $n_s, n_{c}$ | sampling / comparator noise samples (per-instance input unit) | V or uA | sampled in `convert` |

---

- **Internals**: [general internals](../../../internals/analog/adc/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralADCConfig`, `GeneralADCPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
