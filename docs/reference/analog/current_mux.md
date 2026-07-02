# Current mux

## Summary / role

The `CurrentMux` is a leaf analog block: a single-ended N:1 time-share current transport element. It moves one column current at a time through a shared lane, applying a fixed matched transport gain, and time-multiplexes the N columns of a group onto that one lane. It is a structural building block shared across readout chains — one concrete block a consuming circuit composes directly, not a polymorphic family. The block performs the gained transport and tallies the data-dependent rail energy and the serial transport latency; it carries no error hooks.

## Physical model

Physically, an N:1 current mux selects one of N column branches at a time onto a single shared output lane and passes its current through at the lane's transport gain; the group's columns are visited serially, one per transport. The shared lane draws from the rail supply, so the block dissipates over the output current for the duration of each read window, and each serial visit adds a fixed per-operation latency.

This model is ideal: the transport is exact at the configured gain, with no per-lane gain mismatch, no signal-dependent on-resistance, no charge-injection pedestal, no settling error, and no channel-to-channel crosstalk. The design fan-in $N$ (`select_num`) is a structural cross-check against the consuming xbar's reference group size; it does not scale energy or latency. The only modelled effects are the data-dependent rail dissipation and the serial transport latency.

## Governing equations

The lane output current is the gained transport of the selected column current,

$$I_{\mathrm{out}} = g \, I_{\mathrm{in}},$$

with $g$ the dimensionless matched transport gain (`mux_gain`). The rail dissipation over one read window counts the output lane only,

$$E_{\mathrm{dyn}} = V_{\mathrm{supply}} \, |I_{\mathrm{out}}| \, t_{\mathrm{read}},$$

with $V_{\mathrm{supply}}$ the rail voltage (`v_supply__V`) and $t_{\mathrm{read}}$ the read-window width (`read_pulse__ns`). The input current is sourced externally and its production energy is accounted by the upstream block, so it is not counted here. In the consistent unit set $\mathrm{uA} \times \mathrm{V} \times \mathrm{ns} = \mathrm{fJ}$. The serial transport latency scales the per-operation latency by the runtime serial-op count $n_{\mathrm{op}}$,

$$t_{\mathrm{lat}} = t_{\mathrm{op}} \, n_{\mathrm{op}},$$

with $t_{\mathrm{op}}$ the per-transport latency (`latency_per_op__ns`). The serial-op count already tallies the per-group column visits, so the N:1 fan-in is not an extra multiplier; the latency term is logged only when $t_{\mathrm{op}} > 0$.

## Numerical method

N/A — the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

None. The ideal current mux exposes no nonideality sources; its `CurrentMuxPolicy` is an abstract marker with no toggles.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `select_num` ($N$) | design N:1 fan-in; structural cross-check, does not scale energy/latency | — | Design |
| `mux_gain` ($g$) | matched scalar transport gain | — | Design |
| `v_supply__V` ($V_{\mathrm{supply}}$) | rail supply voltage driving the data-dependent dissipation | V | Design |
| `latency_per_op__ns` ($t_{\mathrm{op}}$) | per-transport latency, scaled by the runtime serial-op count | ns | Design |
| `read_pulse__ns` ($t_{\mathrm{read}}$) | read-window width scaling the per-call energy | ns | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the mux is an exact gained-transport time-share element. No per-lane gain mismatch, signal-dependent on-resistance, charge-injection pedestal, settling error, or crosstalk is modelled; the only modelled departures from a lossless ideal are the data-dependent rail dissipation and the serial per-operation latency. The model is valid where the transported current sits within the lane's (unmodelled) linear settled band.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the time-share current transport and the serial-latency model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | selected column input current | uA | `i__uA` |
| $I_{\mathrm{out}}$ | lane output current | uA | `transport` return |
| $g$ | matched transport gain | — | `mux_gain` |
| $N$ | design N:1 fan-in | — | `select_num` |
| $V_{\mathrm{supply}}$ | rail supply voltage | V | `v_supply__V` |
| $t_{\mathrm{op}}$ | per-transport latency | ns | `latency_per_op__ns` |
| $n_{\mathrm{op}}$ | runtime serial-op count | — | derived at logging time |
| $t_{\mathrm{read}}$ | read-window width | ns | `read_pulse__ns` |
| $E_{\mathrm{dyn}}$ | per-call rail dissipation | fJ | logged dynamic energy |
| $t_{\mathrm{lat}}$ | per-call serial latency | ns | logged latency |

---

- **Internals**: [current_mux internals](../../internals/analog/current_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `CurrentMuxConfig`, `CurrentMuxPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
