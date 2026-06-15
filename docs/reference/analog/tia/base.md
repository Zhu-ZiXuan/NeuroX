# TIA Abstract Layer

## Summary / role

Every concrete TIA in the family clamps a bit-line boundary at a virtual-ground reference and absorbs the boundary port current, returning the clamp voltage and its small-signal sensitivity $\partial V_{\mathrm{clamp}}/\partial I_{\mathrm{port}}$ for a current-domain boundary solve. It is one of the two boundary actors a consuming operating-point solve binds, the counterpart of the [driver](../driver.md). This document specifies the shared family contract; the concrete topology is in [opamp_tia](opamp_tia.md).

## Physical model

A TIA holds its input node near a fixed reference voltage by feedback while converting the current it sinks into a clamp voltage. Unlike the ideal [driver](../driver.md) (a zero-output-impedance voltage source), a TIA closes a feedback loop on a virtual-ground reference, so its clamp voltage depends on the port current through a finite, configuration-dependent transfer function. The reference voltage is the family-wide solver-facing constant the abstract layer fixes.

## Governing equations

The clamp transfer function maps the boundary port current to the clamp voltage, supplying the current-domain boundary constraint a consuming operating-point solve binds to,

$$V_{\mathrm{BL,CL}} = \operatorname{TIA}(I_{\mathrm{BL,port}}),$$

and the boundary solve additionally consumes the small-signal sensitivity

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}}$$

(in MOhm), which for an ideal voltage source would be zero but for a finite-gain TIA is non-zero. The transfer function is monotone in the port current so the boundary operating point is unique. The reference voltage $V_{\mathrm{ref}}$ is the virtual-ground level the input is held near; the concrete form of $\operatorname{TIA}(\cdot)$ is topology-specific.

## Numerical method

N/A at the family level - each concrete TIA specifies how its clamp is evaluated; the enclosing operating-point solve is the consumer's, outside this family's scope.

## Noise & non-idealities

The abstract layer fixes no noise source; each concrete TIA declares its own (e.g. op-amp gain mismatch). The TIA's clamp sensitivity entering the solve is the family-wide mechanism by which finite-gain non-idealities perturb the BL boundary.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `v_ref__V` | virtual-ground reference voltage $V_{\mathrm{ref}}$ | V | Design |
| leakage / area / latency | static PPA / spec fields shared by every topology | uW, um^2, ns | Design |

Concrete TIAs add their own topology parameters (see [opamp_tia](opamp_tia.md)). Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the clamp transfer function is monotone in the port current, so the boundary constraint has a unique solution within the consuming operating-point solve.

TODO (domain author): the validity range of the virtual-ground abstraction (input-current range before the loop saturates) common to the family.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{ref}}$ | virtual-ground reference | V | `v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |

---

- **Internals**: [tia base internals](../../../internals/analog/tia/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `TIAConfig` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
