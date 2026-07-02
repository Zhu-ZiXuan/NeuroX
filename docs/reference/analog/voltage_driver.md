# Voltage driver

## Summary / role

The `VoltageDriver` is a generic, design-agnostic boundary clamp: a Thevenin voltage source that holds a crossbar boundary port near a reference voltage and absorbs whatever current the array draws. It is a leaf circuit a column solver composes directly, not a polymorphic family; at $R_{\mathrm{out}} = 0$ it reduces to an ideal constant-voltage source whose held voltage never droops. It supplies one of the two boundary constraints of the array operating-point solve and satisfies the structural clamp-driver role.

## Physical model

The clamp is modelled as a Thevenin equivalent: a reference voltage source $V_{\mathrm{ref}}$ (the open-circuit clamp voltage) in series with a constant output resistance $R_{\mathrm{out}}$. The source sets the held voltage at zero current; the series resistance is the lumped output impedance through which the port current flows, so the clamp voltage droops linearly with the current the clamp sources or sinks. The reality it abstracts is any bounded-impedance drive or sense node — a buffer, a regulated cascode, a clamp transistor in feedback — whose small-signal output impedance over the operating range is summarised by the single constant $R_{\mathrm{out}}$. Setting $R_{\mathrm{out}} = 0$ recovers the ideal voltage source whose held voltage never droops. The reference $V_{\mathrm{ref}}$ is **not** a property the driver self-holds: it is injected per call into `snapshot` as a `Tensor` (the consuming core sources it from a [voltage_reference](voltage_reference.md) and threads the chosen tap in). Two optional non-idealities then perturb that injected reference: a systematic per-instance offset and per-solve thermal noise, both sampled once per read at snapshot time and stored in the snap.

The conduction dissipation that holds the clamp under load — the power $\bigl(V_{\mathrm{supply}} - V_{\mathrm{clamp}}\bigr) I$ burned between the supply rail and the clamped node — is **not** attributed to this block. A Thevenin source is blind to the rail it hangs off; that conduction term flows from the supply the consuming core owns, so the consuming core (the rail owner) tallies it. This block carries only its own static power, folded into its leakage (including any internal amplifier or bias network).

## Governing equations

The clamp transfer function is the Thevenin map

$$V_{\mathrm{clamp}} = \operatorname{driver}(I_{\mathrm{port}}) = V_{\mathrm{ref}} - I_{\mathrm{port}} \, R_{\mathrm{out}},$$

where $V_{\mathrm{ref}}$ is the reference / zero-current clamp voltage (`v_ref__V`, post offset + thermal sample) and $R_{\mathrm{out}}$ is the series output resistance (`r_out__MOhm`). The small-signal output resistance the solver consumes is the constant slope

$$\frac{\partial V_{\mathrm{clamp}}}{\partial I_{\mathrm{port}}} = -R_{\mathrm{out}},$$

reported with units of resistance (MOhm) as the same clamp sensitivity [tia](tia/base.md#governing-equations) carries for its boundary. In the consistent unit set $\mathrm{uA} \times \mathrm{MOhm} = \mathrm{V}$. At $R_{\mathrm{out}} = 0$ the map collapses to the constant $V_{\mathrm{clamp}} = V_{\mathrm{ref}}$ with a zero derivative — the ideal constant-voltage source. This monotone (for $R_{\mathrm{out}} \ge 0$, non-increasing) response composes with the monotone array response so the column operating point is unique; the driver supplies one of the two boundary constraints of the array solve.

## Numerical method

N/A — the transfer function is closed-form (an affine map and its constant derivative); no iteration is involved in evaluating the driver, so it composes directly inside the array operating-point solve in [solver](../xbar/solver.md).

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| reference offset | systematic per-instance mismatch of the reference source | additive zero-mean Gaussian on $V_{\mathrm{ref}}$, state-independent (a single config-constant sigma), sampled per read | offset sigma | `offset` |
| reference thermal noise | thermal fluctuation on the reference node | additive zero-mean Gaussian on $V_{\mathrm{ref}}$, state-independent (a single config-constant sigma), sampled per read | thermal sigma | `thermal` |

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether the thermal term carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `r_out__MOhm` ($R_{\mathrm{out}}$) | series output resistance / constant clamp slope | MOhm | Design |
| offset sigma | Gaussian sigma of the systematic reference offset | V | Measured |
| thermal sigma | Gaussian sigma of the reference thermal noise | V | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. internal amplifier / bias) | uW, um^2 | Design |

The reference / zero-current clamp voltage $V_{\mathrm{ref}}$ is not a config parameter — it is injected per call into `snapshot` as a `Tensor` and carried in the snap (see the [voltage_reference](voltage_reference.md) source that supplies it). Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the clamp is a Thevenin source — a constant reference voltage behind a constant series resistance — so its output impedance is a single value independent of operating point, and its conduction dissipation belongs to the consuming core that owns the supply rail. The model is valid where the lumped $R_{\mathrm{out}}$ captures the node's small-signal output impedance over the operating range (no headroom / compliance limit, no impedance that varies with current).

TODO (domain author): the validity boundary of the constant-$R_{\mathrm{out}}$ idealisation (the current range over which a real clamp's output impedance stays linear), and whether finite slew / settling within the read window is neglected.

## Validation

TODO — link validation evidence once written: that the affine clamp and its constant derivative leave the column operating point well-posed, and that $R_{\mathrm{out}} \to 0$ recovers the ideal driver.

## References

TODO: cite the Thevenin-equivalent clamp model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{clamp}}$ | clamp voltage | V | `solve_clamp` return |
| $V_{\mathrm{ref}}$ | reference / zero-current clamp voltage (injected per call, carried in the snap) | V | `snapshot(v_ref__V=...)`, `VoltageDriverSnap.v_ref__V` |
| $R_{\mathrm{out}}$ | series output resistance | MOhm | `r_out__MOhm` |
| $I_{\mathrm{port}}$ | boundary port current | uA | `i_port__uA` |

---

- **Internals**: [voltage_driver internals](../../internals/analog/voltage_driver.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VoltageDriverConfig`, `VoltageDriverPolicy` (see `api`)
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md)
