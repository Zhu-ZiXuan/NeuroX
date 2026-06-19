# Op-Amp TIA

## Summary / role

Note: this op-amp-TIA spec is incomplete and not yet finalized; its equations are provisional and not to be treated as a settled spec. Open gaps to resolve when finalized: (1) the clamp/output loop equation is not stated, so the clamp transfer function $\operatorname{TIA}$ and its derivative $\partial V_{\mathrm{BL,CL}}/\partial I_{\mathrm{BL,port}}$ cannot be evaluated from the equations given here; (2) the $\tanh$ soft-clip input centering $c = V_{\mathrm{dd}}/2$ inside the argument is a suspected slip — the natural form centers the argument on the input balance point, not on $V_{\mathrm{dd}}/2$; (3) the $s \to 0$ hard-clip limit needs a positive lower bound $s > 0$ to preserve the differentiable rationale of the soft-clip.

`OpAmpTIA` is the op-amp plus NMOS-pseudo-resistor concrete transimpedance amplifier: an operational amplifier closing a feedback loop through an NMOS biased as a pseudo-resistor, clamping the bit line at a virtual-ground reference and converting the column port current into a clamp voltage. It is the only current member of the [TIA family](README.md) and honours the BL clamp-driver contract in [base](base.md).

## Physical model

An op-amp holds its input at the reference voltage $V_{\mathrm{ref}}$ by negative feedback through an NMOS pseudo-resistor (the [access NMOS](../../device/README.md) device model biased into its resistive feedback regime). The finite op-amp gain sets how tightly the input is clamped: a higher gain gives a stiffer virtual ground (smaller clamp-voltage sensitivity to port current), a finite gain leaves a non-zero sensitivity that enters the array solve. The modelled non-ideality is static op-amp gain mismatch; the internal NMOS contributes its own device non-idealities through the owned device.

The real phenomenon at the op-amp output: the output voltage is bounded to the supply window $[0, V_{\mathrm{dd}}]$ and cannot swing past either rail. As the linear drive approaches a rail the output transistors leave their high-gain region and the transfer characteristic soft-saturates, rolling off smoothly into the rail rather than clipping abruptly.

This soft-saturation roll-off is captured by a modeling approximation rather than a transistor-level law: the linear drive is passed through a single-parameter $\tanh$ soft-clip centred at $V_{\mathrm{dd}}/2$ with half-span $V_{\mathrm{dd}}/2$, so the modelled output stays within $[0, V_{\mathrm{dd}}]$ and compresses smoothly toward each rail. The $\tanh$ form is chosen for differentiability and gradient stability, a differentiable modeling choice, not derived from output-stage device physics.

## Governing equations

The clamp transfer function maps the BL port current to the clamp voltage,

$$V_{\mathrm{BL,CL}} = \operatorname{TIA}(I_{\mathrm{BL,port}}),$$

set by the op-amp holding its input near $V_{\mathrm{ref}}$ through the NMOS pseudo-resistor feedback, with the finite-gain virtual-ground stiffness determining the small-signal sensitivity $\partial V_{\mathrm{BL,CL}}/\partial I_{\mathrm{BL,port}}$ (MOhm) the solver consumes. The reference voltage follows the family contract, $V_{\mathrm{ref}} = $ `v_ref__V`.

The op-amp output applies a soft output-rail saturation to the linear drive $x = A\,(V_{\mathrm{ref}} - V_{\mathrm{BL,CL}})$. As a modeling approximation (see Physical model and Assumptions, not a derived transistor-level law), the saturation is taken as the single-parameter $\tanh$ soft-clip

$$V_{\mathrm{out}} = c + h \tanh\!\big((x - c)/s\big), \qquad c = h = \tfrac{1}{2} V_{\mathrm{dd}},\quad s = \texttt{output\_saturation\_softness\_\_V},$$

with the centre and half-span fixed at $c = h = \tfrac{1}{2} V_{\mathrm{dd}}$ so that $V_{\mathrm{out}}$ is confined to $[0, V_{\mathrm{dd}}]$ and compresses smoothly toward each rail; $A$ is the op-amp gain and $s$ the softness scale.

TODO (domain author): write the explicit op-amp + pseudo-resistor transfer function and its derivative in terms of the op-amp gain, the bias voltages, and the NMOS device current, and state how the gain mismatch perturbs the clamp.

## Numerical method

Evaluating the clamp transfer function is topology-specific and is owned by the enclosing operating-point solve, where it enters as a boundary constraint coupling the clamp voltage and the BL port current; the algorithm spec lives in [solver](../../xbar/_1t1r/solver.md).

TODO (domain author): if the clamp evaluation involves an inner solve over the op-amp / NMOS feedback, specify its formulation and well-posedness.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| op-amp gain mismatch | op-amp open-loop gain variation | static Gaussian on the fabricated gain, at fabricate | gain-mismatch sigma | `opamp_gain_sigma` |
| internal NMOS | pseudo-resistor device non-idealities | per the [access NMOS](../../device/README.md) model | NMOS device parameters | `nmos` sub-policy |

The op-amp gain mismatch is static (sampled once per fabricate, per instance); the internal NMOS forwards its own per-source policy through the `nmos` sub-policy.

TODO (domain author): the gain-mismatch sigma's physical derivation and citation.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `v_ref__V` | virtual-ground reference (family field) | V | Design |
| `v_dd__V` | supply rail; sets the soft-clip centre/half-span $V_{\mathrm{dd}}/2$ | V | Design |
| `output_saturation_softness__V` | $\tanh$ output-rail softness scale $s$ | V | Design |
| `opamp_gain` | nominal open-loop gain $A$ | — | Design |
| `v_nmos_bias__V` | pseudo-resistor gate bias | V | Design |
| pseudo-NMOS sizing | feedback NMOS sizing parameters | um | Design |
| bias voltages | op-amp / pseudo-resistor bias | V | Design |
| gain-mismatch sigma | op-amp gain-mismatch sigma | — | Measured |
| owned `NMOSConfig` | the feedback NMOS device config | — | (per device) |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

The owned NMOS device's own parameters are specified in [device](../../device/README.md). Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the feedback element is an NMOS biased as a pseudo-resistor; the clamp stiffness is set by the (finite) op-amp gain.

Stated assumption: the output-rail soft-saturation uses a single-parameter $\tanh$ approximation centred at $V_{\mathrm{dd}}/2$ with half-span $V_{\mathrm{dd}}/2$ and softness $s$. This is a differentiable modeling choice, not the device-specific transfer characteristic; the true output-stage saturation curve depends on the op-amp output topology, sizing, and process, and is not captured beyond the bound $[0, V_{\mathrm{dd}}]$ and a single softness scale.

TODO (domain author): confirm tanh soft-clip is the intended model.

TODO (domain author): the input-current range over which the loop holds the virtual ground, and the regime where the op-amp saturates or the pseudo-resistor leaves its resistive bias.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the op-amp transimpedance clamp and the NMOS pseudo-resistor feedback.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{ref}}$ | virtual-ground reference | V | `v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |
| $V_{\mathrm{out}}$ | soft-saturated op-amp output | V | `v_out__V` |
| $V_{\mathrm{dd}}$ | supply rail (soft-clip centre/half-span $V_{\mathrm{dd}}/2$) | V | `v_dd__V` |
| $s$ | $\tanh$ output-rail softness scale | V | `output_saturation_softness__V` |

---

- **Internals**: [opamp_tia internals](../../../internals/analog/tia/opamp_tia.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `OpAmpTIAConfig`, `OpAmpTIAPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
