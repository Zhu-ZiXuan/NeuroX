# Voltage mux

The voltage mux is a single-ended N:1 time-share transport. Its owner supplies
voltages in an explicit access/lane layout and thereby defines which source
reaches each physical lane during each access. The mux preserves that layout.

## Physical model

Each fabricated lane has a static fractional gain error. Each access also sees
additive voltage noise. The mux records one dynamic-energy item per transported
voltage and derives latency from the number of accesses divided by the number of
fabricated lanes.

## Governing equations

For input shape $[\ldots,A,L]$, where $A$ is the number of serial accesses and
$L$ is the number of parallel lanes, transport requires $A=N$, where $N$ is
`mux_ratio`,

$$
V_{\mathrm{out}}[\ldots,a,l]
= g(1+\varepsilon_{g,l})V_{\mathrm{in}}[\ldots,a,l] + n_{a,l},
$$

where $a$ is the serial access, $l$ is the lane, $g$ is the nominal
transport gain, $\varepsilon_g$ is fabrication-fixed gain mismatch, and $n$ is
per-access additive noise. The output shape is unchanged.

## Numerical method

N/A - the implementation applies elementwise gain and noise without changing
the value tensor's shape.

## Noise & non-idealities

| Source | Statistical model | Parameter |
|---|---|---|
| gain mismatch | static fractional Gaussian mismatch sampled at fabricate | `mux_gain_mismatch_sigma_relative` |
| transport noise | dynamic additive Gaussian noise sampled per access | `mux_noise_sigma__V` |

## Parameters

| Parameter | Meaning | Constraint | Source |
|---|---|---|---|
| `mux_ratio` | N in the N:1 fan-in ratio and required access-axis length | $\geq 1$ | Design |
| `mux_gain` | nominal scalar transport gain | $> 0$ | Design |
| `mux_gain_mismatch_sigma_relative` | per-instance fractional gain-mismatch sigma | $\geq 0$ | Measured |
| `mux_noise_sigma__V` | additive transport-noise sigma | $\geq 0$ | Measured |
| `energy_per_access__fJ` | dynamic energy per transported voltage | $\geq 0$ | Design |
| `latency_per_op__ns` | latency per serial transport round | $\geq 0$ | Design |
| `area_per_inst__um2` | silicon area per physical lane | $\geq 0$ | Design |
| `leakage_per_inst__uW` | leakage per physical lane | $\geq 0$ | Design |

Provenance terms are defined in
[module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

The model is single-ended. It does not model a differential pair, common-mode
rejection, signal-dependent on-resistance, finite settling, charge injection,
clock feedthrough, off-isolation, or crosstalk.

## Validation

TODO - link validation evidence once written.

---

- **Internals**: [voltage_mux internals](../../../internals/primitive/analog/voltage_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `VmuxConfig`, `VmuxPolicy` (see `api`)
