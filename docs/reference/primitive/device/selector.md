# OTS selector

An Ovonic Threshold Switch (OTS) selector is modeled as a static per-cell threshold voltage $V_{\mathrm{th}}$ with additive Gaussian device-to-device mismatch; the model fixes this threshold and its mismatch statistics only and carries no conduction current-voltage law.

## Physical model

The selector is abstracted to a single static per-cell state variable, the threshold voltage $V_{\mathrm{th}}$, shared across all cells at a nominal value before mismatch is applied. Device-to-device variation is modeled as additive Gaussian mismatch sampled once at fabricate time. The model carries no current-voltage relation.

## Governing equations

Each cell $k$ draws its fabricated threshold as the nominal value perturbed by an independent additive Gaussian mismatch:

$$V_{\mathrm{th},k} = V_{\mathrm{th,nom}} + \mathcal{N}_k(0,\,\sigma_{V_{\mathrm{th}}}^2).$$

There is no device-level conduction equation.

## Numerical method

N/A — the model samples a static threshold map and holds no solver or iterative scheme.

## Noise & non-idealities

The single source is parameterised against the shared template in [nonideality](../nonideality.md).

- **$V_{\mathrm{th}}$ mismatch** (`vth_mismatch`, fabricate time) — a state-independent additive zero-mean Gaussian on the nominal threshold with a single constant sigma $\sigma_{V_{\mathrm{th}}}$. This is static device-to-device variation, not per-read noise; its spread carries no area / Pelgrom scaling, unlike the area-matched MOSFET mismatch.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `vth_nominal__V` | nominal threshold voltage $V_{\mathrm{th,nom}}$ | V | TODO (domain author) | Measured |
| `vth_mismatch__V` | additive Gaussian mismatch sigma $\sigma_{V_{\mathrm{th}}}$ | V | $\ge 0$ | Measured |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md). File-level schema: `api`.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{th}}$ | per-cell fabricated threshold voltage | V | `_vth__V` |
| $V_{\mathrm{th,nom}}$ | nominal threshold voltage | V | `config.vth_nominal__V` |
| $\sigma_{V_{\mathrm{th}}}$ | mismatch sigma | V | `vth_mismatch__V` |

## Assumptions, scope & validity

- The selector is reduced to a threshold-voltage map; no conduction I-V law, hysteresis, or holding behavior is modeled at the device level.
- Mismatch is static (sampled at fabricate time), additive, and Gaussian; no per-read threshold noise is modeled.
- The threshold is temperature-independent at this level.

TODO (domain author): give the validity range of the Gaussian mismatch assumption, and whether OTS conduction / temperature dependence must be modeled for the studies of interest.

## Validation

TODO: link `validation/device` evidence — threshold mismatch-statistics checks against measured selector data.

## References

TODO: cite the OTS selector device and its threshold-mismatch characterization.

---

- **Internals**: [selector internals](../../../internals/primitive/device/selector.md)
- **Validation**: TODO — `validation/device` (not yet written)
- **Configuration**: `api` (`SelectorConfig`, `SelectorPolicy`)
