# Current mux

## Physical model

An ideal single-ended N:1 time-share current transport with a matched gain across all lanes.

## Governing equations

For every supplied current value,

$$I_{\mathrm{out}} = g\,I_{\mathrm{in}}.$$

Transport has no modeled energy or latency.

## Numerical method

N/A — closed-form map.

## Noise & non-idealities

None.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `mux_ratio` ($N$) | N in the physical N:1 fan-in ratio | — | $\geq 1$ | Design |
| `mux_gain` ($g$) | matched scalar transport gain | — | $> 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $I_{\mathrm{in}}$ | single-ended column-current input | uA | `i__uA` |
| $I_{\mathrm{out}}$ | transported current | uA | `transport` return |
| $g$ | matched transport gain | — | `mux_gain` |
| $N$ | N:1 fan-in ratio | — | `mux_ratio` |

## Assumptions, scope & validity

The model applies within each lane's linear settled band. It excludes per-lane gain mismatch, signal-dependent on-resistance, charge-injection pedestal, settling error, and crosstalk.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the time-share current transport model.
