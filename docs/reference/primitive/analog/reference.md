# Reference source

## Physical model

A reference source holds a scalar or rectangular array of nominal values. Each fabricated instance receives one fixed relative initial-accuracy error per value. The values' units and axis meanings belong to the circuit that uses them; the source preserves their configured shape without selecting among them.

## Governing equation

For every configured value $x_j^{\mathrm{nom}}$,

$$x_j = x_j^{\mathrm{nom}}(1 + \delta_j), \qquad \delta_j \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2).$$

The error is sampled once at fabrication and remains fixed for the instance lifetime. A nominal zero remains exactly zero.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `values` | nominal scalar or rectangular value array | owner-defined | non-empty rectangular tuple tree or scalar | Design |
| `tolerance_sigma_relative` | relative initial-accuracy sigma | — | $\geq 0$ | Measured |
| leakage / area | standing power and silicon area | uW, um² | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions and scope

The bias topology, load-dependent droop, access-varying noise, drift, and selection energy are outside this model. Standing bias power is represented by the static PPA field and is independent of the configured values.
