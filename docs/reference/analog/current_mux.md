# Current mux

## Physical model

An ideal single-ended N:1 time-share current transport: the shared output lane carries one selected column current at a time, scaled by an exact matched transport gain, and the group's N columns are visited serially, one transport each. It is a pure current-transport primitive — it draws no rail energy of its own (the downstream current-domain consumer that owns the rail tallies dissipation); beyond the exact transported value the model tallies one effect, the fixed serial per-operation latency, which the N:1 fan-in does not scale.

## Governing equations

The lane output current is the gained transport of the selected column current,

$$I_{\mathrm{out}} = g \, I_{\mathrm{in}},$$

with $g$ the dimensionless matched transport gain. The serial latency scales the per-operation latency by the serial-op count $n_{\mathrm{op}}$,

$$t_{\mathrm{lat}} = t_{\mathrm{op}} \, n_{\mathrm{op}},$$

with $t_{\mathrm{op}}$ the per-transport latency and $n_{\mathrm{op}}$ the number of serial column visits in the group. The serial-op count already tallies those visits, so the N:1 fan-in is not a further multiplier.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

None. The transport is ideal; the neglected non-idealities are named in Assumptions, scope & validity.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `select_num` ($N$) | design N:1 fan-in (columns sharing the lane); does not scale latency | — | $\geq 1$ | Design |
| `mux_gain` ($g$) | matched scalar transport gain | — | $> 0$ | Design |
| `latency_per_op__ns` ($t_{\mathrm{op}}$) | per-transport latency, scaled by the serial-op count | ns | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | selected column input current | uA | `i__uA` |
| $I_{\mathrm{out}}$ | lane output current | uA | `transport` return |
| $g$ | matched transport gain | — | `mux_gain` |
| $N$ | N:1 fan-in (columns per group) | — | `select_num` |
| $t_{\mathrm{op}}$ | per-transport latency | ns | `latency_per_op__ns` |
| $n_{\mathrm{op}}$ | serial-op count (column visits per group) | — | derived at logging time |
| $t_{\mathrm{lat}}$ | serial latency over one group | ns | logged latency |

## Assumptions, scope & validity

The mux is modelled as an exact gained-transport time-share element. No per-lane gain mismatch, signal-dependent on-resistance, charge-injection pedestal, settling error, or crosstalk is modelled; the only departure from a lossless ideal is the serial per-operation latency, and it accounts no rail energy of its own. The model is valid where the transported current sits within the lane's (unmodelled) linear settled band.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the time-share current transport and the serial-latency model.

---

- **Internals**: [current_mux internals](../../internals/analog/current_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `CurrentMuxConfig`, `CurrentMuxPolicy` (see `api`)
