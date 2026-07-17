# Op-amp TIA

The op-amp TIA is the non-linear, closed-loop transimpedance amplifier of the [TIA family](family.md): an operational amplifier closes a feedback loop through an NMOS biased as a pseudo-resistor, clamping the bit line at a virtual-ground reference $V_{\mathrm{ref}}$ (a per-call input) and converting the column port current into a clamp voltage.

## Physical model

An op-amp holds its input at the reference voltage $V_{\mathrm{ref}}$ by negative feedback through an NMOS pseudo-resistor — an NMOS biased into its resistive feedback regime, modelled by the [MOSFET](../../device/mosfet.md) device model. The finite op-amp gain sets how tightly the input is clamped: a higher gain gives a stiffer virtual ground (smaller clamp-voltage sensitivity to port current), a finite gain leaves a non-zero sensitivity. The modelled non-idealities are static op-amp gain mismatch and the internal NMOS device non-idealities.

The op-amp output is bounded to the supply window $[0, V_{\mathrm{dd}}]$. The model captures this saturation with a single-parameter $\tanh$ soft-clip centred at $V_{\mathrm{dd}}/2$ with half-span $V_{\mathrm{dd}}/2$, so the output stays within $[0, V_{\mathrm{dd}}]$ and compresses smoothly toward each rail.

## Governing equations

The clamp transfer function maps the BL port current to the clamp voltage,

$$V_{\mathrm{BL,CL}} = \operatorname{TIA}(I_{\mathrm{BL,port}}),$$

set by the op-amp holding its input near $V_{\mathrm{ref}}$ through the NMOS pseudo-resistor feedback, with the finite-gain virtual-ground stiffness determining the small-signal sensitivity $\partial V_{\mathrm{BL,CL}}/\partial I_{\mathrm{BL,port}}$ (MOhm).

The op-amp output applies a soft output-rail saturation to the linear drive $x = A\,(V_{\mathrm{ref}} - V_{\mathrm{BL,CL}})$, taken as the single-parameter $\tanh$ soft-clip

$$V_{\mathrm{out}} = c + h \tanh\!\big((x - c)/s\big), \qquad c = h = \tfrac{1}{2} V_{\mathrm{dd}},$$

with the centre and half-span fixed at $c = h = \tfrac{1}{2} V_{\mathrm{dd}}$ so that $V_{\mathrm{out}}$ is confined to $[0, V_{\mathrm{dd}}]$ and compresses smoothly toward each rail; $A$ is the op-amp gain and $s$ the softness scale.

The clamp operating point satisfies the bit-line KCL constraint that the NMOS pseudo-resistor current equals the BL port current, with residual $f = I(V_{\mathrm{BL,CL}}) - I_{\mathrm{BL,port}}$. Its Jacobian with respect to the clamp voltage is

$$\frac{\partial f}{\partial V_{\mathrm{BL,CL}}} = \frac{\partial I}{\partial v_d}\,(-A\,g_{\mathrm{clip}}) + \frac{\partial I}{\partial v_s},$$

with the NMOS drain and source held at $v_d = V_{\mathrm{out}}$ and $v_s = V_{\mathrm{BL,CL}}$, where $\partial I/\partial v_d$ and $\partial I/\partial v_s$ are the NMOS drain- and source-voltage transconductances, $A$ is the op-amp gain, and $g_{\mathrm{clip}}$ is the local $\tanh$ soft-clip gradient. By the implicit function theorem the small-signal clamp sensitivity is the reciprocal of this Jacobian (since $\partial f/\partial I_{\mathrm{BL,port}} = -1$):

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}} = \left(\frac{\partial f}{\partial V_{\mathrm{BL,CL}}}\right)^{-1}.$$

TODO (domain author): write the explicit closed-form op-amp + pseudo-resistor transfer function (the loop is currently given implicitly by the KCL root) and state how the gain mismatch perturbs the clamp.

## Numerical method

The clamp voltage is the root of the scalar KCL residual $f(V_{\mathrm{BL,CL}}) = I(V_{\mathrm{BL,CL}}) - I_{\mathrm{BL,port}} = 0$ (Governing equations): a one-dimensional root-find in $V_{\mathrm{BL,CL}}$, solved by Newton iteration with the Jacobian $\partial f/\partial V_{\mathrm{BL,CL}}$ above. Under the monotone-clamp assumption (Assumptions) the residual is monotone in $V_{\mathrm{BL,CL}}$, so the root is unique.

TODO (domain author): the convergence guarantee for the Newton iteration and the input-current range over which the loop stays in the monotone regime.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| op-amp gain mismatch | op-amp open-loop gain variation | static Gaussian on the fabricated gain, at fabrication | gain-mismatch sigma |
| internal NMOS | pseudo-resistor device non-idealities | per the [MOSFET](../../device/mosfet.md) device model | NMOS device parameters |

The op-amp gain mismatch is static, fixed per instance at fabrication; the internal NMOS contributes its own device non-idealities per the [MOSFET](../../device/mosfet.md) device model.

TODO (domain author): the gain-mismatch sigma's physical derivation and citation.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `v_dd__V` | supply rail; sets the soft-clip centre/half-span $V_{\mathrm{dd}}/2$ | V | $> 0$ | Design |
| `output_saturation_softness__V` | $\tanh$ output-rail softness scale $s$ | V | $> 0$ | Design |
| `opamp_gain` | nominal open-loop gain $A$ | — | $> 1$ | Design |
| `v_nmos_bias__V` | pseudo-resistor gate bias | V | TODO (domain author) | Design |
| pseudo-NMOS sizing | feedback NMOS sizing parameters | um | $> 0$ | Design |
| gain-mismatch sigma | op-amp gain-mismatch sigma | — | $\geq 0$ | Measured |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | $\geq 0$ | Design |

The virtual-ground reference $V_{\mathrm{ref}}$ is a per-call runtime input, not a config parameter. The internal NMOS device's own parameters are specified in the [MOSFET](../../device/mosfet.md) device model. Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | virtual-ground reference (per-call input) | V | `snapshot(v_ref__V=...)`, `OpAmpTiaSnap.v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | `i_port__uA` |
| $I$ | NMOS pseudo-resistor current | uA | `ids__uA` |
| $V_{\mathrm{out}}$ | soft-saturated op-amp output | V | `v_out__V` |
| $V_{\mathrm{dd}}$ | supply rail (soft-clip centre/half-span $V_{\mathrm{dd}}/2$) | V | `v_dd__V` |
| $A$ | op-amp open-loop gain | — | `opamp_gain` |
| $x$ | op-amp linear drive $A(V_{\mathrm{ref}} - V_{\mathrm{BL,CL}})$ | V | `v_out_lin` |
| $c$ | soft-clip centre, $\tfrac{1}{2}V_{\mathrm{dd}}$ | V | `softclip_center__V` |
| $h$ | soft-clip half-span, $\tfrac{1}{2}V_{\mathrm{dd}}$ | V | `softclip_half_span__V` |
| $s$ | $\tanh$ output-rail softness scale | V | `output_saturation_softness__V` |
| $g_{\mathrm{clip}}$ | local $\tanh$ soft-clip gradient | — | `g_clip` |
| $f$ | clamp-node KCL residual | uA | `residual__uA` |
| $v_d$ | NMOS drain voltage, $= V_{\mathrm{out}}$ | V | `vd__V` |
| $v_s$ | NMOS source voltage, $= V_{\mathrm{BL,CL}}$ | V | `vs__V` |

## Assumptions, scope & validity

Stated assumption: the feedback element is an NMOS biased as a pseudo-resistor; the clamp stiffness is set by the (finite) op-amp gain.

Stated assumption: the output-rail soft-saturation uses a single-parameter $\tanh$ approximation centred at $V_{\mathrm{dd}}/2$ with half-span $V_{\mathrm{dd}}/2$ and softness $s$. This is a differentiable modeling choice, not the device-specific transfer characteristic; the true output-stage saturation curve depends on the op-amp output topology, sizing, and process, and is not captured beyond the bound $[0, V_{\mathrm{dd}}]$ and a single softness scale.

TODO (domain author): confirm the $\tanh$ soft-clip is the intended output-rail model, in particular whether its argument centres on $V_{\mathrm{dd}}/2$ or on the input balance point.

TODO (domain author): the input-current range over which the loop holds the virtual ground, and the regime where the op-amp saturates or the pseudo-resistor leaves its resistive bias.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the op-amp transimpedance clamp and the NMOS pseudo-resistor feedback.

---

- **Internals**: [opamp_tia internals](../../../../internals/primitive/analog/tia/opamp_tia.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `OpAmpTiaConfig`, `OpAmpTiaPolicy` (see `api`)
