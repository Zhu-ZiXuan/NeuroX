# OTS Selector

## Summary

`Selector` models an Ovonic Threshold Switch (OTS) selector as a per-cell threshold-voltage map with static fabrication mismatch. In the device→circuit→architecture stack it is a device-layer primitive that publishes a per-cell threshold voltage $V_{\mathrm{th}}$; how that threshold gates conduction (the switching law itself) is decided by the consuming circuit. This document specifies the threshold value domain and its mismatch statistics; the selector owns no conduction I-V law of its own at this level.

## Physical model

The selector is abstracted to a single static per-cell state variable, the threshold voltage $V_{\mathrm{th}}$, shared across all cells at a nominal value before mismatch is applied. Device-to-device variation is modeled as additive Gaussian mismatch sampled once per fabricate call onto the per-instance threshold map. The model carries no current-voltage relation: the threshold is exported to the consuming circuit, which owns the switching behavior built on top of it.

## Governing equations

Each cell $k$ draws its fabricated threshold as the nominal value perturbed by an independent additive Gaussian mismatch:

$$V_{\mathrm{th},k} = V_{\mathrm{th,nom}} + \mathcal{N}_k(0,\,\sigma_{V_{\mathrm{th}}}^2).$$

There is no device-level conduction equation.

## Numerical method

N/A — the model samples a static threshold map and exports it; it holds no solver and no iterative scheme.

## Noise & non-idealities

The single source is parameterised against the shared template in [notation_conventions](../notation_conventions.md#noise-model-conventions).

- **$V_{\mathrm{th}}$ mismatch** (`vth_mismatch`, fabricate time) — a state-independent additive zero-mean Gaussian on the nominal threshold with a single config-constant sigma $\sigma_{V_{\mathrm{th}}}$ (no area dependence), sampled once per `fabricate()` call. Switched by the per-run policy flag; with the flag off the map is the uniform nominal value. This is static device-to-device variation, not per-read noise. Area / Pelgrom scaling of the spread is deliberately not modelled here, unlike the [NMOS](nmos.md#noise--non-idealities) mismatch.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `vth_nominal__V` | nominal threshold voltage $V_{\mathrm{th,nom}}$ | V | Measured |
| `vth_mismatch__V` | additive Gaussian mismatch sigma $\sigma_{V_{\mathrm{th}}}$ | V | Measured |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md). File-level schema: `api`.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{th}}$ | per-cell fabricated threshold voltage | V | `vth__V` |
| $V_{\mathrm{th,nom}}$ | nominal threshold voltage | V | `nominal_vth__V` |
| $\sigma_{V_{\mathrm{th}}}$ | mismatch sigma | V | `vth_mismatch__V` |

## Assumptions, scope & validity

Stated assumptions of the current model:

- The selector is reduced to a threshold-voltage map; no conduction I-V law, hysteresis, or holding behavior is modeled at the device level.
- Mismatch is static (sampled at fabricate time), additive, and Gaussian; no per-read threshold noise is modeled.
- The threshold is temperature-independent at this level: an operating temperature is supplied at construction but does not enter the model.

TODO (domain author): state the intended use of the threshold map by the consuming circuit, the validity range of the Gaussian mismatch assumption, and whether OTS conduction / temperature dependence must be modeled for the studies of interest.

## Validation

TODO: link `validation/device` evidence — threshold mismatch-statistics checks against measured selector data.

## References

TODO: cite the OTS selector device and its threshold-mismatch characterization.

---

- **Internals**: [selector internals](../../internals/device/selector.md)
- **Validation**: TODO — `validation/device` (not yet written)
- **Configuration**: `api` (`SelectorConfig`, `SelectorPolicy`)
- **Decisions**: N/A — no ADR governs this device.
