# SL Driver

## Summary / role

The `Driver` is the ideal constant-voltage source-line (SL) clamp-driver: it holds the SL boundary of a crossbar column at a fixed clamp voltage and absorbs whatever port current the array draws. It is the SL-side boundary actor in the array DC operating point, the counterpart of the BL-side [tia](tia/README.md). It is a leaf circuit a column solver composes directly, not a polymorphic family.

## Physical model

The driver is modelled as an ideal voltage source: its output voltage is independent of the current it sources or sinks. The reality it abstracts is a strong drive buffer whose output impedance is negligible against the SL wire and access-device impedances over the operating range; the abstraction takes that output impedance as exactly zero. The single optional non-ideality is additive thermal noise on the held voltage, sampled once per read.

## Governing equations

The clamp transfer function is the constant map

$$V_{\mathrm{SL,CL}} = \operatorname{driver}_{\mathrm{SL}}(I_{\mathrm{SL,port}}) = V_{\mathrm{drive}} + n,$$

independent of the port current $I_{\mathrm{SL,port}}$, where $n$ is the per-read noise sample ($n = 0$ when the thermal-noise policy is off). The small-signal output resistance the solver consumes is therefore identically zero,

$$\frac{\partial V_{\mathrm{SL,CL}}}{\partial I_{\mathrm{SL,port}}} = 0,$$

with units of resistance (MOhm), the same clamp sensitivity [tia](tia/base.md#governing-equations) carries for the BL boundary.

This constant, monotone (degenerately flat) response composes with the monotone array response so the column operating point is unique; the driver supplies one of the two boundary constraints of the array solve specified in [circuit_core](../xbar/_1t1r/circuit_core.md#governing-equations).

## Numerical method

N/A - the transfer function is closed-form (a constant and its zero derivative); no iteration is involved in evaluating the driver. The enclosing array operating-point solve is in [solver](../xbar/solver.md).

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| drive thermal noise | thermal fluctuation on the buffered drive node | additive zero-mean Gaussian on $V_{\mathrm{SL,CL}}$, state-independent (a single config-constant sigma) | drive-thermal sigma | `drive_thermal` |

TODO (domain author): give the noise sigma's physical derivation and citation, and confirm whether it is white per read or carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `drive_value` | nominal SL clamp voltage $V_{\mathrm{drive}}$ | V | Design |
| drive-thermal sigma | Gaussian noise sigma on the held voltage | V | Measured |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the driver is an ideal voltage source with zero output impedance over the full operating range, so its delivered voltage never droops under load.

TODO (domain author): the validity boundary of the zero-output-impedance idealisation (maximum port current before a real buffer droops), and whether finite slew / settling within the WL pulse is neglected.

## Validation

TODO - link validation evidence once written: that the constant clamp and its zero derivative leave the column operating point well-posed.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{SL,CL}}$ | SL clamp voltage | V | `v_sl_drive` |
| $V_{\mathrm{drive}}$ | nominal drive voltage | V | `drive_value` |
| $I_{\mathrm{SL,port}}$ | SL boundary port current | uA | derived from node voltages |
| $n$ | per-read thermal-noise sample | V | sampled in snap |

---

- **Internals**: [driver internals](../../internals/analog/driver.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `DriverConfig` (see `api`)
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md)
