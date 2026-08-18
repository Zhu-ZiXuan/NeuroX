# General voltage ADC

The simplest member of the [differential voltage ADC family](family.md): a single-mode digitizer whose resolution is fixed by its comparator count, obeying the family raw-code and floor-quantization contract.

## Physical model

A flat bank of $n_{\mathrm{codes}} - 1$ comparators. The model floors the differential input against their threshold voltages; the bucket index is the raw code. Two additive Gaussian stages perturb the signal before bucketization. The comparator count is the topology and is fixed at construction, while the thresholds are the injected reference taps, one ascending 1-D ladder supplied per call and shared by every input position.

## Governing equations

The conversion maps the raw differential input $x^{+} - x^{-}$ directly to a raw unsigned code. With input-referred sampling noise $n_s$ and comparator noise $n_c$,

$$\mathrm{code} = \operatorname{clamp}\!\Big(\operatorname{bucketize}\big(x^{+}-x^{-}+n_s+n_{c},\ \{B_c\}\big),\ 0,\ n_{\mathrm{codes}}-1\Big),$$

where $\operatorname{bucketize}$ floors the signal against the injected thresholds $\{B_c\} = V_{\mathrm{ref}}$ and $n_{\mathrm{codes}}$ is the configured code count. The raw code lies in $[0,\ n_{\mathrm{codes}}-1]$ and its signed magnitude is obtained by subtracting the topology zero code $z$. The underlying floor against ordered boundaries is the [family quantization law](family.md#governing-laws).

The topology has one code-count-implied resolution $b$.

## Numerical method

N/A — the conversion is a single floor-bucketize evaluation per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| sampling noise | input-referred sample jitter | additive zero-mean Gaussian on the input | `sampling_noise__V` |
| comparator noise | comparator (thermal/decision) noise | single additive zero-mean Gaussian on the signal | `comparator_noise__V` |
| quantization | intrinsic floor-bucketize | floor against the injected ordered thresholds | `v_refs__V` (per call) |

All sources are dynamic, sampled per conversion; the model carries no static mismatch.

TODO (domain author): physical derivation and citation for each noise sigma.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `code_num` ($n_{\mathrm{codes}}$) | output code count; the comparator count is one less | — | $\geq 2$ | Design |
| `sampling_noise__V` | input-referred sampling-noise sigma | V | $\geq 0$ | Measured |
| `comparator_noise__V` | comparator-noise sigma on the post-transform signal | V | $\geq 0$ | Measured |
| `energy_per_op__fJ` | dynamic energy per conversion | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | per-conversion latency | ns | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

The comparator thresholds are not parameters of this ADC — they are the reference taps supplied per conversion (see [family](family.md)), calibrated against physical data by the owner via the ADC calibration procedure ([calibration guide](../../../../guides/calibration/README.md)). Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $x^{+}, x^{-}$ | differential input legs (per-instance input unit) | V or uA | `v_pos__V`, `v_neg__V` |
| $B_c$ | comparator threshold at index $c$ (per-instance input unit) | V or uA | `v_refs__V` (per call) |
| $n_{\mathrm{codes}}$ | number of code buckets | — | `code_num` |
| $z$ | topology zero code | — | `zero_code` / `zero_offset(bits)` |
| $b$ | code-count-implied resolution (bits) | — | `max_bits` |
| $n_s, n_{c}$ | sampling / comparator noise samples (per-instance input unit) | V or uA | sampled in `convert` |

## Assumptions, scope & validity

Stated assumptions:

- The bit width is fixed by the code count.
- The injected ladder is 1-D, carries exactly $n_{\mathrm{codes}} - 1$ taps, and is sorted, so the bucketize order is preserved.

TODO (domain author): the validity boundary of the behavioural-comparator model and the input-range limits implied by the threshold list.

## Validation

TODO - link validation evidence once written.

## References

TODO.
