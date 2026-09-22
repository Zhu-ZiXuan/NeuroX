# OTS selector

## Physical model

An Ovonic Threshold Switch (OTS) selector is represented by a static per-cell threshold $V_{\mathrm{th}}$ with a common nominal value and additive Gaussian fabrication mismatch.

## Governing equations

Each cell $k$ receives an independent fabricated threshold:

$$V_{\mathrm{th},k} = V_{\mathrm{th,nom}} + \mathcal{N}_k(0,\,\sigma_{V_{\mathrm{th}}}^2).$$

## Numerical method

N/A — direct sampling of static thresholds.

## Noise & non-idealities

Threshold mismatch follows the additive law in [nonideality](../nonideality.md), with constant sigma $\sigma_{V_{\mathrm{th}}}$ and no area scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `vth_nominal__V` | nominal threshold voltage $V_{\mathrm{th,nom}}$ | V | TODO (domain author) | Measured |
| `vth_mismatch__V` | additive Gaussian mismatch sigma $\sigma_{V_{\mathrm{th}}}$ | V | $\ge 0$ | Measured |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $V_{\mathrm{th}}$ | per-cell fabricated threshold voltage | V | `_vth__V` |
| $V_{\mathrm{th,nom}}$ | nominal threshold voltage | V | `config.vth_nominal__V` |
| $\sigma_{V_{\mathrm{th}}}$ | mismatch sigma | V | `vth_mismatch__V` |

## Assumptions, scope & validity

- Conduction I-V, hysteresis, and holding behavior are excluded.
- No per-read threshold noise is modeled.
- The threshold is temperature-independent at this level.

TODO (domain author): give the validity range of the Gaussian mismatch assumption, and whether OTS conduction / temperature dependence must be modeled for the studies of interest.

## Validation

TODO: link `validation/device` evidence — threshold mismatch-statistics checks against measured selector data.

## References

TODO: cite the OTS selector device and its threshold-mismatch characterization.
