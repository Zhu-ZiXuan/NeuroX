# OTS selector

## Physical model

An Ovonic Threshold Switch (OTS) selector is represented by a static per-cell threshold $V_{\mathrm{th}}$ with a common nominal value and additive Gaussian fabrication mismatch.

## Governing equations

Each cell $k$ receives an independent fabricated threshold:

$$V_{\mathrm{th},k} = V_{\mathrm{th,nom}} + \mathcal{N}_k(0,\,\sigma_{V_{\mathrm{th}}}^2).$$

## Noise & non-idealities

Threshold mismatch follows the additive law in [nonideality](../nonideality.md), with constant sigma $\sigma_{V_{\mathrm{th}}}$ and no area scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | [Source](../../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| `vth_nominal__V` | nominal threshold voltage $V_{\mathrm{th,nom}}$ | V | Not established | Measured |
| `vth_mismatch__V` | additive Gaussian mismatch sigma $\sigma_{V_{\mathrm{th}}}$ | V | $\ge 0$ | Measured |

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

The measured validity range of Gaussian threshold mismatch is not established. This model supplies thresholds rather than an OTS conduction or temperature model.

## Validation

Threshold-mismatch statistics have not been compared with measured selector data in this document.

## References

Citations for the OTS selector device and its threshold-mismatch characterization are not documented here.
