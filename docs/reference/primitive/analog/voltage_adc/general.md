# General voltage ADC

The simplest member of the [voltage ADC family](family.md): a single-mode digitizer whose resolution is fixed by a calibrated list of comparator thresholds, obeying the family raw-code and floor-quantization contract.

## Physical model

The model floors the differential input against a sorted list of voltage thresholds; the bucket index is the raw code. A monotone input transform - identity or base-2 logarithm - is applied before bucketization. Two additive Gaussian stages perturb the signal: input-referred sampling noise before the transform and comparator noise after it. The bucketization uses no reference tap.

## Governing equations

The conversion maps the raw differential input $x^{+} - x^{-}$ to a raw unsigned code. A monotone input transform $g$ - the identity (linear) or the base-2 logarithm (logarithmic companding) - is applied to the noisy input before the bucketize; being monotone it preserves the boundary ordering. With input-referred sampling noise $n_s$ added before the transform and comparator noise $n_{c}$ after it,

$$\mathrm{code} = \operatorname{clamp}\!\Big(\operatorname{bucketize}\big(g(x^{+}-x^{-}+n_s)+n_{c},\ \{B_c\}\big),\ 0,\ n_{\mathrm{codes}}-1\Big),$$

where $\operatorname{bucketize}$ floors the transformed signal against the fixed thresholds $\{B_c\}$ and $n_{\mathrm{codes}}$ is the number of code buckets implied by the threshold list. In the linear case the signal and thresholds are in volts; in the logarithmic case $g$ floors its argument at $\epsilon = 10^{-12}$ V before the base-2 logarithm. The raw code lies in $[0,\ n_{\mathrm{codes}}-1]$ and its signed magnitude is obtained by subtracting the topology zero code $z$. The underlying floor against ordered boundaries is the [family quantization law](family.md#governing-laws).

The topology has one boundary-implied resolution $b$.

## Numerical method

N/A - the conversion is a single floor-bucketize evaluation per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| sampling noise | input-referred sample jitter | additive zero-mean Gaussian on the input | `sampling_noise__V` |
| comparator noise | comparator (thermal/decision) noise | single additive zero-mean Gaussian on the signal | `comparator_noise__V` |
| quantization | intrinsic floor-bucketize | floor against ordered thresholds | `boundaries` |

All sources are dynamic, sampled per conversion; the model carries no static mismatch.

TODO (domain author): physical derivation and citation for each noise sigma.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `boundaries` ($B_c$) | sorted comparator thresholds, in the instance input unit | uA or V | strictly increasing | Calibrated (physical data) |
| `input_transform` ($g$) | input companding law | — | linear or log2 | Design |
| `sampling_noise__V` | input-referred sampling-noise sigma | V | $\geq 0$ | Measured |
| `comparator_noise__V` | comparator-noise sigma on the post-transform signal | V | $\geq 0$ | Measured |
| `energy_per_op__fJ` | dynamic energy per conversion | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | per-conversion latency | ns | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md). The comparator thresholds are calibrated against physical data via the ADC calibration procedure ([calibration guide](../../../../guides/calibration/README.md)).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $g$ | monotone input transform (identity or $\log_2$) | — | `input_transform` |
| $B_c$ | comparator threshold at index $c$ (per-instance input unit) | V or uA | `boundaries` |
| $n_{\mathrm{codes}}$ | number of code buckets | — | derived from `boundaries` |
| $z$ | topology zero code | — | `zero_code` / `zero_offset(bits)` |
| $b$ | boundary-implied resolution (bits) | — | `max_bits` |
| $n_s, n_{c}$ | sampling / comparator noise samples (per-instance input unit) | V or uA | sampled in `convert` |

## Assumptions, scope & validity

Stated assumptions:

- The bit width is fixed by the threshold list.
- The threshold list is sorted, so the bucketize order is preserved.

TODO (domain author): the validity boundary of the behavioural-comparator model and the input-range limits implied by the threshold list.

## Validation

TODO - link validation evidence once written.

## References

TODO.

---

- **Internals**: [general internals](../../../../internals/primitive/analog/voltage_adc/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralDifferentialVoltageAdcConfig`, `GeneralDifferentialVoltageAdcPolicy` (see `api`)
