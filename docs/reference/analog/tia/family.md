# TIA family

## Summary / role

Every concrete TIA in the family is the BL clamp-driver: it clamps a bit-line boundary at a virtual-ground reference and absorbs the boundary port current, returning the clamp voltage and its small-signal sensitivity $\partial V_{\mathrm{clamp}}/\partial I_{\mathrm{port}}$ for a current-domain boundary solve. It supplies one of the two boundary constraints of the array operating-point solve, the counterpart of the SL clamp-driver voltage_driver. This document specifies the shared family contract; each concrete topology is in its own document.

## Physical model

A TIA holds its input node near a fixed reference voltage by feedback while converting the current it sinks into a clamp voltage. Unlike the voltage_driver (a Thevenin clamp, ideal at zero output impedance), a TIA closes a feedback loop on a virtual-ground reference, so its clamp voltage depends on the port current through a finite, configuration-dependent transfer function. The reference voltage is not a TIA constant: it is injected per call into `snapshot` as a `Tensor` (the consuming core sources it from a [voltage_reference](../voltage_reference.md)).

## Governing equations

The clamp transfer function maps the boundary port current to the clamp voltage, supplying the current-domain boundary constraint at the BL port,

$$V_{\mathrm{BL,CL}} = \operatorname{TIA}(I_{\mathrm{BL,port}}),$$

and the boundary solve additionally consumes the small-signal sensitivity

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}}$$

(in MOhm), which for an ideal clamp-driver (the infinite-gain, zero-input-impedance limit) would be zero but for a finite-gain TIA is non-zero. A monotone clamp transfer function is necessary but not sufficient for a unique boundary operating point: uniqueness of the coupled fixed point also requires the array-side response to be monotone in a compatible direction, so the monotone clamp composes with the monotone array response to a single intersection (mirroring the SL-side condition on the voltage_driver). The reference voltage $V_{\mathrm{ref}}$ is the virtual-ground level the input is held near; the concrete form of $\operatorname{TIA}(\cdot)$ is topology-specific.

## Numerical method

N/A at the family level - each concrete TIA specifies how its clamp is evaluated; the enclosing operating-point solve lies outside this family's scope.

## Noise & non-idealities

The abstract layer fixes no noise source; each concrete TIA declares its own (e.g. op-amp gain mismatch). The TIA's clamp sensitivity entering the solve is the family-wide mechanism by which finite-gain non-idealities perturb the BL boundary.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| leakage / area | static PPA fields shared by every topology | uW, um^2 | Design |

The virtual-ground reference voltage $V_{\mathrm{ref}}$ is not a config parameter — it is injected per call into `snapshot` as a `Tensor` and carried in the snap (see the [voltage_reference](../voltage_reference.md) source). Concrete TIAs add their own topology parameters. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the clamp transfer function is monotone in the port current. This is the family's share of the uniqueness condition; a unique boundary operating point additionally requires the compatible array-side monotonicity stated in §Governing equations.

TODO (domain author): the validity range of the virtual-ground abstraction (input-current range before the loop saturates) common to the family.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | virtual-ground reference (injected per call, carried in the snap) | V | `snapshot(v_ref__V=...)`, `*Snap.v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |

---

- **Internals**: [tia base internals](../../../internals/analog/tia/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `TIAConfig` (see `api`)
