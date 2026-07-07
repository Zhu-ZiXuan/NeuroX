# General ADC

## Summary / role

`GeneralADC` is the boundary-bucketize behavioural ADC: a sorted list of comparator thresholds plus additive Gaussian noise stages. It is the simplest member of the [ADC family](family.md) - a single-mode digitizer whose resolution is fixed by its boundary list. It honours the family signed-code and floor contract in [base](family.md).

## Physical model

The ADC is modelled as a bank of comparators against a sorted threshold list `boundaries` (in input units - uA for current-mode, V for voltage-mode). A monotone input transform - the identity for a linear ADC or a base-2 logarithm (companding) for a logarithmic one - maps the differential input before it is compared against every threshold; the index of the bucket it falls into is the raw code. The model has two additive Gaussian noise stages - input-referred sampling noise applied before the transform and a single comparator-noise term applied after it. The bucketize is reference-free: it uses no reference tap.

## Governing equations

The conversion maps the raw differential input $x^{+} - x^{-}$ to a signed code. A monotone input transform $g$ - the identity (linear) or the base-2 logarithm (logarithmic companding) - is applied to the noisy input before the bucketize; being monotone it preserves the boundary ordering. With input-referred sampling noise $n_s$ added before the transform and comparator noise $n_{c}$ after it,

$$\mathrm{code} = \operatorname{clamp}\!\Big(\operatorname{bucketize}\big(g(x^{+}-x^{-}+n_s)+n_{c},\ \{B_c\}\big),\ 0,\ n_{\mathrm{codes}}-1\Big) - z,$$

where $\operatorname{bucketize}$ floors the transformed noisy signal against the fixed boundaries $\{B_c\}$, $n_{\mathrm{codes}}$ is the number of code buckets implied by the boundary list, and $z$ is the topology zero code. In the linear case the signal and boundaries share the per-instance input unit (uA for current-mode, V for voltage-mode). The signed output lies in $[-z,\ n_{\mathrm{codes}}-1-z]$. This is the deterministic floor-bucketize law of the [family contract](family.md#governing-equations); the clamp catches a bucket index pushed out of range by noise before the zero shift. The calibrated `boundaries` may be non-uniformly spaced and need not number a power of two, consistent with the generalized base contract.

The topology is single-mode: its one operating point is $\mathrm{mode} = 0$ at the boundary-implied bit width $b$.

## Numerical method

N/A - the conversion is a single floor-bucketize evaluation per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| sampling noise | input-referred sample jitter | additive zero-mean Gaussian on the input | `sampling_noise__V` |
| comparator noise | comparator (thermal/decision) noise | single additive zero-mean Gaussian on the signal | `comparator_noise__V` |
| quantization | intrinsic floor-bucketize | deterministic floor against ordered boundaries | `boundaries` |

All sources are dynamic (sampled per convert); `GeneralADC` carries no static mismatch.

TODO (domain author): physical derivation and citation for each noise sigma.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `boundaries` | sorted comparator thresholds (input units) | uA or V | Calibrated (physical data) |
| `input_transform` | input companding law (linear or log2) | — | Design |
| `sampling_noise__V` | input-referred sampling-noise sigma | V | Measured |
| `comparator_noise__V` | input-referred comparator-noise sigma | V | Measured |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md). The comparator thresholds are calibrated against physical data via the ADC calibration procedure ([calibration guide](../../../guides/calibration/README.md)).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $g$ | monotone input transform (identity or $\log_2$) | — | `input_transform` |
| $B_c$ | comparator threshold at index $c$ (per-instance input unit) | V or uA | `boundaries` |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived from `boundaries` |
| $z$ | topology zero code | — | `_zero_code` |
| $n_s, n_{c}$ | sampling / comparator noise samples (per-instance input unit) | V or uA | sampled in `convert` |

## Assumptions, scope & validity

Stated assumptions:

- The bit width is fixed by the boundary list (single-mode).
- The `boundaries` are sorted, so the bucketize order is preserved.

TODO (domain author): the validity boundary of the behavioural-comparator model and the input-range limits implied by the boundary list.

## Validation

TODO - link validation evidence once written.

## References

TODO.

---

- **Internals**: [general internals](../../../internals/analog/adc/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralADCConfig`, `GeneralADCPolicy` (see `api`)
