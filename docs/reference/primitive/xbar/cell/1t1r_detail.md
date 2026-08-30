# 1T1R Detail cell

The Detail 1T1R cell realizes the [1T1R family topology](1t1r.md) with nonlinear device models: one RRAM device in series with an EKV access NMOS, $\mathrm{BL} - \mathrm{RRAM} - V_{\mathrm{X}} - \mathrm{NMOS} - \mathrm{SL}$. It condenses the access node by a per-cell Newton solve of the internal KCL.

## Physical model

The RRAM conducts between the bit line $V_{\mathrm{BL}}$ and $V_{\mathrm{X}}$; the access NMOS conducts between $V_{\mathrm{X}}$ and the source line $V_{\mathrm{SL}}$, gated by the word-line voltage $V_{\mathrm{WL}}$. The EKV access-NMOS model is source/drain symmetric, so source and drain labels are a naming convention: the SL-side terminal ($V_{\mathrm{SL}}$) is taken as the source and the internal BL-side terminal ($V_{\mathrm{X}}$) as the drain, with no effect on the device current. The word-line drive and the device read state are inputs fixed per call. The device transfer functions are specified in [RRAM](../../device/rram.md) and [MOSFET](../../device/mosfet.md).

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ the access node satisfies the internal KCL — the RRAM current into $V_{\mathrm{X}}$ equals the NMOS current out of it,

$$F_{\mathrm{X}} = I_{\mathrm{N}}\!\left(V_{\mathrm{WL}}, V_{\mathrm{X}}, V_{\mathrm{SL}}\right) - I_{\mathrm{R}}\!\left(V_{\mathrm{BL}} - V_{\mathrm{X}}\right) = 0.$$

Solving this for $V_{\mathrm{X}}$ condenses the cell. The condensed branch then reports a **single consistent current**: at convergence $I_{\mathrm{R}} = I_{\mathrm{N}}$, and that common value is the branch current $I$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$). The residual $\lvert I_{\mathrm{N}} - I_{\mathrm{R}} \rvert$ at the condensed $V_{\mathrm{X}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** of its condensed branch, obtained by eliminating $V_{\mathrm{X}}$ from the series stack — the series condensation of the NMOS drain conductance and the RRAM conductance at the access node,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} = \frac{g_{\mathrm{N},d}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} = \frac{g_{\mathrm{N},s}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \le 0,$$

where $g_{\mathrm{R}} = \partial I_{\mathrm{R}}/\partial(V_{\mathrm{BL}} - V_{\mathrm{X}}) \ge 0$ is the RRAM differential conductance and $g_{\mathrm{N},d} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{X}} \ge 0$, $g_{\mathrm{N},s} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{SL}} \le 0$ are the NMOS drain / source partials. The signs follow from the device I-V laws: the RRAM and NMOS-drain conductances are non-negative and the NMOS-source partial is non-positive, so $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$ for any operating point.

## Numerical method

The internal KCL $F_{\mathrm{X}}(V_{\mathrm{X}}) = 0$ is strictly monotone in $V_{\mathrm{X}}$: $I_{\mathrm{N}}$ is increasing in $V_{\mathrm{X}}$ and $I_{\mathrm{R}}$ is decreasing in $V_{\mathrm{X}}$, so $F_{\mathrm{X}} = I_{\mathrm{N}} - I_{\mathrm{R}}$ is strictly increasing and has a unique root for any $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$. The root is found by a Pade current-divider seed — a first-order split of the BL-to-SL drop across the NMOS output conductance and the programmed RRAM conductance — followed by a fixed number of Newton steps on $F_{\mathrm{X}}$. The step count is a calibrated numerical knob (see §Parameters); convergence is fast because the seed lands near the root and the residual decreases geometrically once in the Newton regime.

## Noise & non-idealities

The cell introduces no static mismatch of its own; non-idealities enter through its two devices: [RRAM](../../device/rram.md) conductance non-idealities and access-[MOSFET](../../device/mosfet.md) threshold / transconductance mismatch. The read state is fixed per call, so the device noise is sampled once per call and the condensation is deterministic given that state.

## Parameters

In addition to the [shared family parameters](1t1r.md):

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `rram_config` | RRAM storage-device configuration | — | — | see [reference/device/rram](../../device/rram.md) |
| `nmos_config` | access-NMOS configuration | — | — | see [reference/device/mosfet](../../device/mosfet.md) |
| `state_to_g_map__uS` | state-index to target conductance | uS | strictly increasing; length $\ge 2$; endpoints in $[G_{\mathrm{min}}, G_{\mathrm{RRAM,max}}]$ | Calibrated (physical data) |
| `access_nmos_W__um` | access-NMOS width | um | $> 0$ | Design |
| `access_nmos_L__um` | access-NMOS length | um | $> 0$ | Design |
| `rram_g_max__uS` | maximum programmable RRAM conductance | uS | $> G_{\mathrm{min}}$ | Design |
| `newton_iter_num` | per-cell access-node Newton iteration count | — | $\ge 1$ | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md); how to obtain values for a new chip: [calibration guide](../../../../guides/calibration/README.md). A config file selects this model with `_neurox_class = "XbarCell1t1rDetailConfig"` in the cell table and its policy file with `XbarCell1t1rDetailPolicy`; the `_neurox_class` directive and the file-level schema are specified in [configuration](../../../../api/configuration.md).

## Energy model

The Detail cell exposes the shared family node levels, with $V_{\mathrm{X}}$ from its per-cell Newton condensation; it assigns them no capacitance or energy.

## Symbols

In addition to the [shared family symbols](1t1r.md):

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $I_{\mathrm{R}}$ | RRAM current | uA | `Rram.solve_dc` |
| $I_{\mathrm{N}}$ | access-NMOS current | uA | `Nmos.solve_dc` |
| $g_{\mathrm{R}}$ | RRAM differential conductance | uS | `Rram.solve_dc` |
| $g_{\mathrm{N},d}, g_{\mathrm{N},s}$ | NMOS drain / source partials | uS | `Nmos.solve_dc` |
| $G_{\mathrm{RRAM,max}}$ | max programmable RRAM conductance | uS | `rram_g_max__uS` |
| $G_{\mathrm{min}}$ | RRAM device conductance floor | uS | `rram_config.g_min__uS` |

## Assumptions, scope & validity

- The series stack is RRAM then access NMOS; the condensation solves the exact nonlinear device stack at each operating point.
- The solve is quasi-static: it finds the DC access-node operating point and does not model transient device switching within a pulse.

## Validation

TODO: add validation evidence for branch-current and signed-conductance checks, internal-KCL residuals, and finite-difference device derivatives.

## References

TODO: cite the RRAM and access-NMOS current models and the series-condensation basis.
