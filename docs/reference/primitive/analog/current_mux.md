# Current mux

## Physical model

An ideal single-ended N:1 time-share current transport. The surrounding circuit arranges the serial accesses and parallel lanes before calling the mux; this primitive applies the matched transport gain elementwise without changing the supplied layout.

## Governing equations

For every supplied current value,

$$I_{\mathrm{out}} = g\,I_{\mathrm{in}}.$$

The primitive models no energy and reports a zero window. It neither groups an input axis nor validates the caller's access layout.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

None. The transport is ideal; the neglected non-idealities are named in Assumptions, scope & validity.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `mux_ratio` ($N$) | N in the physical N:1 fan-in ratio | — | $\geq 1$ | Design |
| `mux_gain` ($g$) | matched scalar transport gain | — | $> 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended column-current input | uA | `i__uA` |
| $I_{\mathrm{out}}$ | transported current | uA | `transport` return |
| $g$ | matched transport gain | — | `mux_gain` |
| $N$ | N:1 fan-in ratio | — | `mux_ratio` |

## Assumptions, scope & validity

The mux is modelled as an exact gained-transport time-share element. Axis grouping and scheduling belong to the surrounding circuit. No per-lane gain mismatch, signal-dependent on-resistance, charge-injection pedestal, settling error, or crosstalk is modelled. The model is valid where the transported current sits within the lane's unmodelled linear settled band.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the time-share current transport model.
