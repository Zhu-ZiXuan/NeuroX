# Op-amp TIA

The concrete op-amp TIA: it owns an internal `NMOS`, fabricates its own op-amp gain, and extends the family clamp surface with a closed-loop `solve_dc`.

## Design decisions

- **Owns the feedback `NMOS` directly.** `OpAmpTIAConfig` carries the owned `MOSFETConfig`; `__init__` constructs the internal NMOS with the same `inst_shape` and forwards `policy.nmos` into it - the parent constructs the child, no external factory closure. Because `NMOS` inherits `FabricateMixin`, `OpAmpTIA.fabricate()` auto-cascades into it, so the op-amp's own `_sample_fabricate_mismatch` only refreshes `opamp_gain`.
- **Composite policy.** `OpAmpTIAPolicy(TIAPolicy)` is structured: a leaf `opamp_gain_sigma: bool` plus a nested `nmos: MOSFETPolicy` sub-policy forwarded to the owned device. The nesting mirrors the ownership tree, so each device's switches travel with it.
- **Inner damped-Newton solve.** `solve_dc` runs a fixed number of step-damped Newton iterations on the clamp-node KCL `I_nmos(v_clamp) - i_port = 0` for a single port current, returning the converged `v_clamp__V`, its small-signal sensitivity `dVclamp_dI__MOhm`, the companion op-amp output `v_out__V` and `dVout_dI__MOhm`, and the final-iterate KCL residual `residual__uA`. The iteration count `config.n_newton` is a compile-time constant picked by that solver calibrator, which detects the plateau over the `v_clamp__V` iterate sequence (when successive clamp updates stop shrinking); the returned `residual__uA` is the secondary relative-residual sanity guard. Method-intrinsic step caps `MAX_STEP__V` and `G_EFF_MAX__uS` (see Gotchas) keep the iteration from diverging or sticking at a rail and are deliberately not exposed in the config.

## Contracts & invariants

- **Concrete snap and the `solve_dc` extension.** `OpAmpTIA` fixes the family snapshot / solve_clamp surface ([base](base.md)) to `OpAmpTIASnap`, whose per-call fields are the injected `v_ref__V` plus the fabricated `opamp_gain` and the owned `nmos_snap`. Beyond that surface it adds `solve_dc(...) -> OpAmpTIADCOP`, returning `v_out__V`, `dVout_dI__MOhm`, and `residual__uA` - a concrete extension over the base surface.
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
