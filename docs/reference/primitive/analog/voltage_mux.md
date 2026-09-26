# Voltage mux

## Physical model

A single-ended N:1 time-share voltage transport with static fractional gain error per instance, additive noise per access, and fixed dynamic energy per transported voltage.

## Governing equations

$$
V_{\mathrm{out}}
= g(1+\varepsilon_g)V_{\mathrm{in}} + n,
$$

Here $g$ is the nominal gain, $\varepsilon_g$ the fabricated gain mismatch, and $n$ the access noise.

## Noise & non-idealities

| [Source](../../../conventions/module_parameter.md) | Statistical model | Parameter |
| --- | --- | --- |
| gain mismatch | static fractional Gaussian mismatch sampled at fabricate | `mux_gain_mismatch_sigma_relative` |
| transport noise | dynamic additive Gaussian noise sampled per access | `mux_noise_sigma__V` |

## Parameters

| Parameter | Meaning | Constraint | [Source](../../../conventions/module_parameter.md) |
| --- | --- | --- | --- |
| `mux_ratio` | N in the physical N:1 fan-in ratio | $\geq 1$ | Design |
| `mux_gain` | nominal scalar transport gain | $> 0$ | Design |
| `mux_gain_mismatch_sigma_relative` | per-instance fractional gain-mismatch sigma | $\geq 0$ | Measured |
| `mux_noise_sigma__V` | additive transport-noise sigma | $\geq 0$ | Measured |
| `energy_per_access__fJ` | dynamic energy per transported voltage | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per physical lane | $\geq 0$ | Design |
| `leakage_per_inst__uW` | leakage per physical lane | $\geq 0$ | Design |

## Assumptions, scope & validity

The model excludes differential-pair behavior, common-mode rejection, signal-dependent on-resistance, finite settling, charge injection, clock feedthrough, off-isolation, and crosstalk.
