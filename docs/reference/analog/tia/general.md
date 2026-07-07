# General TIA

## Summary / role

`GeneralTIA` is the linear concrete transimpedance amplifier: a Thevenin-input plus resistive-transimpedance model that clamps the bit line at a virtual-ground reference and converts the column port current into a clamp voltage and an output voltage by two constant resistances. It is the linear, loop-free sibling of opamp_tia and honours the BL clamp-driver contract in [base](family.md).

## Physical model

The clamp node is modelled as a Thevenin source: a virtual-ground reference $V_{\mathrm{ref}}$ behind a small-signal input impedance $Z_{\mathrm{in}}$. The port current flowing into that impedance lifts the clamp node off the ideal reference by a linear drop, so a larger $Z_{\mathrm{in}}$ gives a softer virtual ground. The conversion stage is an equivalent load resistance $R_{\mathrm{load}}$ that maps the port current to the output voltage about the same reference — the transimpedance gain. Both resistances are constant, so the transfer is exact and carries no nonideality.

## Governing equations

The clamp node and the output are affine in the port current $I_{\mathrm{BL,port}}$ about the reference $V_{\mathrm{ref}}$,

$$V_{\mathrm{BL,CL}} = V_{\mathrm{ref}} + Z_{\mathrm{in}} \, I_{\mathrm{BL,port}}, \qquad V_{\mathrm{out}} = V_{\mathrm{ref}} + R_{\mathrm{load}} \, I_{\mathrm{BL,port}},$$

with $Z_{\mathrm{in}} = $ `input_impedance__MOhm` and $R_{\mathrm{load}} = $ `load_resistance__MOhm`. The small-signal sensitivities the solver consumes are the constants

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}} = Z_{\mathrm{in}}, \qquad \frac{\partial V_{\mathrm{out}}}{\partial I_{\mathrm{BL,port}}} = R_{\mathrm{load}}.$$

## Numerical method

The transfer is closed-form and exact — no inner solve. Evaluating the clamp constraint enters the enclosing operating-point solve as a linear boundary coupling the clamp voltage and the BL port current.

## Noise & non-idealities

None. The model is ideal: both resistances are deterministic constants.

## Energy model

The transimpedance-resistor dissipation over one read window is

$$E = R_{\mathrm{load}} \, I_{\mathrm{BL,port}}^{2} \, t_{\mathrm{read}},$$

with $t_{\mathrm{read}}$ the read-window width.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `input_impedance__MOhm` | Thevenin small-signal input impedance $Z_{\mathrm{in}}$ | MOhm | Design |
| `load_resistance__MOhm` | conversion-stage load resistance $R_{\mathrm{load}}$ (transimpedance gain) | MOhm | Design |
| leakage / area | static PPA fields | uW, um^2 | Design |

The virtual-ground reference $V_{\mathrm{ref}}$ is not a config parameter — it is injected per call into `snapshot` as a `Tensor` and carried in `GeneralTIASnap.v_ref__V` (see the [voltage_reference](../voltage_reference.md) source). Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | virtual-ground reference (injected per call, carried in the snap) | V | `snapshot(v_ref__V=...)`, `GeneralTIASnap.v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |
| $V_{\mathrm{out}}$ | transimpedance output | V | `v_out__V` |
| $Z_{\mathrm{in}}$ | Thevenin input impedance | MOhm | `input_impedance__MOhm` |
| $R_{\mathrm{load}}$ | conversion-stage load resistance | MOhm | `load_resistance__MOhm` |
| $t_{\mathrm{read}}$ | read-window width | ns | `read_pulse__ns` (xbar) |

## Assumptions, scope & validity

Stated assumption: the clamp node and the conversion stage are linear over the operating range — the input impedance and the load resistance are constant and the output does not saturate against a supply rail. The model carries no rail bound and no soft-clip; outside the linear range (where a real output stage would compress toward a rail) use opamp_tia.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the resistive transimpedance clamp.

---

- **Internals**: TODO - internals doc not yet written.
- **Validation**: TODO - validation evidence not yet written.
- **Configuration**: `GeneralTIAConfig`, `GeneralTIAPolicy` (see `api`)
