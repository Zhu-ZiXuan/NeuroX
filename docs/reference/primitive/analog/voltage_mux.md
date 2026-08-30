# Voltage mux

The voltage mux is a single-ended N:1 time-share transport. The surrounding circuit arranges the serial accesses and parallel lanes before calling it.

## Physical model

Each fabricated instance has a static fractional gain error. Each transported voltage also sees additive noise. The mux records one dynamic-energy item per transported voltage. It applies these effects elementwise to the layout supplied by the surrounding circuit.

## Governing equations

$$
V_{\mathrm{out}}
= g(1+\varepsilon_g)V_{\mathrm{in}} + n,
$$

where $g$ is the nominal transport gain, $\varepsilon_g$ is fabrication-fixed gain mismatch, and $n$ is per-call additive noise. The primitive neither groups an input axis nor validates the caller's access layout.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

| Source | Statistical model | Parameter |
|---|---|---|
| gain mismatch | static fractional Gaussian mismatch sampled at fabricate | `mux_gain_mismatch_sigma_relative` |
| transport noise | dynamic additive Gaussian noise sampled per access | `mux_noise_sigma__V` |

## Parameters

| Parameter | Meaning | Constraint | Source |
|---|---|---|---|
| `mux_ratio` | N in the physical N:1 fan-in ratio | $\geq 1$ | Design |
| `mux_gain` | nominal scalar transport gain | $> 0$ | Design |
| `mux_gain_mismatch_sigma_relative` | per-instance fractional gain-mismatch sigma | $\geq 0$ | Measured |
| `mux_noise_sigma__V` | additive transport-noise sigma | $\geq 0$ | Measured |
| `energy_per_access__fJ` | dynamic energy per transported voltage | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per physical lane | $\geq 0$ | Design |
| `leakage_per_inst__uW` | leakage per physical lane | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

The model is single-ended. Axis grouping and scheduling belong to the surrounding circuit. It does not model a differential pair, common-mode rejection, signal-dependent on-resistance, finite settling, charge injection, clock feedthrough, off-isolation, or crosstalk.

## Validation

TODO - link validation evidence once written.
