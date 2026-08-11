# Current mux

## Physical model

An ideal single-ended N:1 time-share current transport. Its owner supplies currents in an explicit access/lane layout and thereby defines which source reaches each physical lane during each access. The mux preserves that layout and applies an exact matched transport gain.

## Governing equations

For an input with shape $[\ldots,A,L]$, where $A$ is the number of serial accesses and $L$ is the number of parallel lanes, transport requires $A=N$, where $N$ is `mux_ratio`, and applies

$$I_{\mathrm{out}}[\ldots,a,l] = g\,I_{\mathrm{in}}[\ldots,a,l].$$

The output has the same shape. Axis $a$ indexes time-serial accesses and axis $l$ indexes spatially parallel lanes. The primitive emits no energy and reports a zero window; the consuming readout owns those costs.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

None. The transport is ideal; the neglected non-idealities are named in Assumptions, scope & validity.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `mux_ratio` ($N$) | N in the N:1 fan-in ratio and required access-axis length | — | $\geq 1$ | Design |
| `mux_gain` ($g$) | matched scalar transport gain | — | $> 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended column-current input | uA | `i__uA` |
| $I_{\mathrm{out}}$ | access/lane output | uA | `transport` return |
| $g$ | matched transport gain | — | `mux_gain` |
| $N$ | N:1 fan-in ratio | — | `mux_ratio` |

## Assumptions, scope & validity

The mux is modelled as an exact gained-transport time-share element. No per-lane gain mismatch, signal-dependent on-resistance, charge-injection pedestal, settling error, or crosstalk is modelled. The model is valid where the transported current sits within the lane's (unmodelled) linear settled band.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the time-share current transport model.

---

- **Internals**: [current_mux internals](../../../internals/primitive/analog/current_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `ImuxConfig`, `ImuxPolicy` (see `api`)
