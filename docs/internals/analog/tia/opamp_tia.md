# Op-amp TIA

## Summary

`OpAmpTIA` (`tia/opamp_tia.py`) is the op-amp plus NMOS-pseudo-resistor concrete TIA: it owns an internal `NMOS`, fabricates op-amp gain state, and threads runtime clamp snaps through the solver-facing protocol.

## Design decisions

- **Owns the feedback `NMOS` directly.** `OpAmpTIAConfig` carries the owned `MOSFETConfig`; `__init__` constructs the internal NMOS with the same `inst_shape` and forwards `policy.nmos` into it - the parent constructs the child, no external factory closure. Because `NMOS` inherits `FabricateMixin`, `OpAmpTIA.fabricate()` auto-cascades into it, so the op-amp's own `_sample_fabricate_mismatch` only refreshes `opamp_gain`.
- **Composite policy.** `OpAmpTIAPolicy(TIAPolicy)` is structured: a leaf `opamp_gain_sigma: bool` plus a nested `nmos: MOSFETPolicy` sub-policy forwarded to the owned device. The nesting mirrors the ownership tree, so each device's switches travel with it.
- **Inner damped-Newton solve.** `solve_dc` solves the clamp-node KCL `I_nmos(v_clamp) - i_port = 0` for a single port current and returns the converged clamp voltage `v_clamp__V`, its small-signal sensitivity `dVclamp_dI__MOhm` (with the companion op-amp-output voltage and sensitivity), and the final-iterate KCL residual `residual__uA`. A scheme's solver calibrator drives plateau detection over the iteration count from the `v_clamp__V` iterate sequence (when successive clamp updates stop shrinking); the `residual__uA` is the secondary relative-residual sanity guard. The one chip-facing knob is `config.n_newton`, the fixed number of step-damped Newton iterations and a compile-time constant picked by that calibrator. Method-intrinsic step caps `MAX_STEP__V` and `G_EFF_MAX__uS` (named constants, see Gotchas) keep the iteration from diverging or sticking at a rail; they are not exposed in the config.

## Contracts & invariants

- **Concrete snap and the `solve_dc` extension.** `OpAmpTIA` fixes the family's snapshot / solve_clamp surface ([base](base.md)) to `OpAmpTIASnap`, whose per-call fields are the injected `v_ref__V` plus the fabricated `opamp_gain` and the owned `nmos_snap`. Beyond that surface it adds `solve_dc(...) -> OpAmpTIADCOP`, which also returns the op-amp output voltage, its sensitivity, and the KCL residual - the concrete extension the array solve does not depend on.
- **Fabricate cascade.** The op-amp gain is fabricated state on the op-amp; the NMOS state is fabricated through the owned child. A single `fabricate()` refreshes both.

## Performance & resources

TODO - record the clamp-evaluation cost and any inner-solve memory once profiled.

## Gotchas

- **Do not bypass the owned NMOS's policy.** Device non-idealities reach the NMOS only through `policy.nmos`; setting op-amp switches without the sub-policy leaves the feedback device at its own defaults.
- **`MAX_STEP__V` is a class-level Newton step cap (method-intrinsic, not chip-tuneable).** It caps $|\Delta v_{\mathrm{clamp}}|$ per iteration. Without it, an overshoot near a rail - where `tanh`'s `g_clip → 0` - sends `v_clamp` to the rail in one shot, and the next iterate computes `f/f'` at the rail, typically trapping the solver there.
- **`G_EFF_MAX__uS` is the upper clamp on the effective KCL Jacobian `df/dV_clamp` (always negative).** Without it the Newton step diverges when the NMOS feedback loop's local derivative crosses zero. Like `MAX_STEP__V` it is a method-intrinsic constant, not exposed in the config, because it does not vary per chip preset.

## Known limitations

- **Clamp transfer function not yet specified in Reference.** The explicit closed-form op-amp + pseudo-resistor transfer function is a domain-author TODO; verification coverage of the clamp against an independent reference is correspondingly a gap.

---

- **Reference**: [opamp_tia](../../../reference/analog/tia/opamp_tia.md)
- **Implementation**: `neurox/analog/tia/opamp_tia.py`
- **Tests**: TODO - name the guarding test
