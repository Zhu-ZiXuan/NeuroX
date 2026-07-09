# General TIA

## Physical model

The clamp node is modelled as a Thevenin source: a virtual-ground reference $V_{\mathrm{ref}}$ behind a small-signal input impedance $Z_{\mathrm{in}}$. The port current flowing into that impedance lifts the clamp node off the ideal reference by a linear drop, so a larger $Z_{\mathrm{in}}$ gives a softer virtual ground. The conversion stage is an equivalent load resistance $R_{\mathrm{load}}$ that maps the port current to the output voltage about the same reference — the transimpedance gain. Both resistances are constant, so the transfer is exact and carries no nonideality.

## Governing equations

The clamp node and the output are affine in the port current $I_{\mathrm{BL,port}}$ about the reference $V_{\mathrm{ref}}$,

$$V_{\mathrm{BL,CL}} = V_{\mathrm{ref}} + Z_{\mathrm{in}} \, I_{\mathrm{BL,port}}, \qquad V_{\mathrm{out}} = V_{\mathrm{ref}} + R_{\mathrm{load}} \, I_{\mathrm{BL,port}},$$

with constant input impedance $Z_{\mathrm{in}}$ and load resistance $R_{\mathrm{load}}$. The small-signal sensitivities are the constants

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}} = Z_{\mathrm{in}}, \qquad \frac{\partial V_{\mathrm{out}}}{\partial I_{\mathrm{BL,port}}} = R_{\mathrm{load}}.$$

The transimpedance-resistor dissipation over one read window is

$$E = R_{\mathrm{load}} \, I_{\mathrm{BL,port}}^{2} \, t_{\mathrm{read}},$$

with $t_{\mathrm{read}}$ the read-window width. In the consistent unit set $\mathrm{uA}^2 \times \mathrm{MOhm} \times \mathrm{ns} = \mathrm{fJ}$.

## Numerical method

Closed-form and exact — no inner solve.

## Noise & non-idealities

None. The model is ideal: both resistances are deterministic constants.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `input_impedance__MOhm` ($Z_{\mathrm{in}}$) | Thevenin small-signal input impedance | MOhm | $\geq 0$ | Design |
| `load_resistance__MOhm` ($R_{\mathrm{load}}$) | conversion-stage load resistance (transimpedance gain) | MOhm | $> 0$ | Design |
| leakage / area | static PPA fields | uW, um^2 | $\geq 0$ | Design |

The virtual-ground reference $V_{\mathrm{ref}}$ is a supplied input, not a config parameter; a [voltage_reference](../voltage_reference.md) is the source that sets it. Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | virtual-ground reference (injected per call, carried in the snap) | V | `snapshot(v_ref__V=...)`, `GeneralTIASnap.v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | `i_port__uA` |
| $V_{\mathrm{out}}$ | transimpedance output | V | `v_out__V` |
| $Z_{\mathrm{in}}$ | Thevenin input impedance | MOhm | `input_impedance__MOhm` |
| $R_{\mathrm{load}}$ | conversion-stage load resistance | MOhm | `load_resistance__MOhm` |
| $t_{\mathrm{read}}$ | read-window width | ns | `read_pulse__ns` |
| $E$ | transimpedance-resistor dissipation over one read window | fJ | `dynamic_energy__fJ` return |

## Assumptions, scope & validity

The clamp node and the conversion stage are assumed linear over the operating range: the input impedance and the load resistance are constant and the output does not saturate against a supply rail. The model carries no rail bound and no soft-clip; outside this range, where a real output stage would compress toward a rail, the transfer is unmodelled.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the resistive transimpedance clamp.

---

- **Internals**: TODO - internals doc not yet written.
- **Validation**: TODO - validation evidence not yet written.
- **Configuration**: `GeneralTIAConfig`, `GeneralTIAPolicy` (see `api`)
