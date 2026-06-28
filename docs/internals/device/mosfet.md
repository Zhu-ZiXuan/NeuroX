# MOSFET — Implementation

## Summary

`MOSFET` (`mosfet.py`) is a stateless-in-conduction electrical primitive: it precomputes temperature-scaled nominals and Pelgrom sigmas at construction, resamples per-cell `beta__uA_per_V2` / `vth__V` at fabricate time, and evaluates the EKV-softplus I-V and its three node partials in closed form. The abstract base carries all physics; the concrete `NMOS` / `PMOS` subclasses fix only the channel polarity. It realizes the model in [reference/device/mosfet](../../reference/device/mosfet.md). This document covers the non-obvious choices, not the I-V flow.

## Design decisions

- **MOSFET is a pure electrical primitive — no layout parasitics.** Layout-dependent lumped capacitances are deliberately excluded; the placed-transistor parasitics belong to the circuit that places the transistor, which carries its own design parameters and computes its own effective-capacitance scalars. Pushing them into the device would couple a foundry primitive to a specific cell layout.
- **`W__um` / `L__um` are `__init__` kwargs, not config fields.** Channel geometry is a device design parameter chosen per placement, so it stays out of the frozen `MOSFETConfig` (which holds process electricals and the Pelgrom matching coefficients only) per the project rule.
- **Polarity lives in the concrete class, not in config.** The channel polarity is a class attribute of `NMOS` (`+1`) / `PMOS` (`-1`); the shared data classes `MOSFETConfig` / `MOSFETPolicy` / `MOSFETSnap` / `MOSFETDCOP` are polarity-free. `beta__uA_per_V2` stays a positive magnitude and the polarity factor enters the I-V law once (in `I_ds = 0.5 · p · β · (v_s² − v_d²)`), squaring out of the terminal partials so the `did_dvd >= 0` / `did_dvs <= 0` contract is polarity-independent. The base raises if instantiated directly or if `polarity` is not `±1`.
- **Nominals and Pelgrom sigmas are precomputed once at construction.** Temperature scaling ($\mu(T)$, $V_{\mathrm{th}}(T)$), the softplus smoothing scale $1/(2 n V_T)$, and the area-scaled sigmas $\sigma_{V_{\mathrm{th}}}$, $\sigma_\beta$ depend only on init-time constants, so they are computed once and stored, not recomputed per solve. The `0.1` factor in $\beta_{\mathrm{nom}}$ is the cm^2→um^2 / unit reconciliation that lands the result in uA/V^2.
- **Policy is separate and per-call-site constructed.** `MOSFETPolicy` (two flat `bool` mismatch flags, no defaults) is loaded from its own file; with a flag off the corresponding `apply_gaussian` is the identity, so noise-off runs allocate no mismatch draw.

## Contracts & invariants

- **`_sample_fabricate_mismatch()` is re-callable and shape-stable.** Each call re-expands the scalar nominals to `self._inst_shape` and resamples; the owning module drives the cadence via `model.fabricate()` and the mixin auto-cascades. Fabricated `beta__uA_per_V2` / `vth__V` live on the device and are not mirrored elsewhere.
- **`snapshot(shape, multi_coords)` is the read path into fabricated state.** It expands the per-cell maps to the per-call `shape`, optionally advanced-indexes a chunk via `multi_coords`, and returns a `MOSFETSnap`. The solve reads `beta__uA_per_V2` and `vth__V` only from the snap, never from `self`.
- **`solve_dc(vg, vd, vs, snap)` returns three node partials with fixed signs.** $\partial I/\partial V_d \ge 0$ and $\partial I/\partial V_s \le 0$ by construction for both polarities; the consuming Jacobian relies on these signs. Inputs may be scalars or tensors and broadcast against the snap.

## Performance & resources

State is two per-cell buffers at `inst_shape` plus their scalar nominals. The solve is allocation-light: softplus / sigmoid on the two overdrives, no iteration. The smoothing scale and sigmas are scalars, so they do not scale with `inst_shape`.

## Gotchas

- **The softplus uses PyTorch's `beta` as the smoothing scale, not as $\beta$.** `F.softplus(x, beta=inv_smooth_scale)` reuses the framework keyword `beta` for the sharpness $\lambda = 1/(2 n V_T)$; it is unrelated to the transconductance factor `beta__uA_per_V2`. Do not conflate the two.
- **`T__K` enters at construction, not per call.** Temperature scaling and the smoothing scale are baked into the precomputed constants; changing temperature requires reconstruction, not a per-solve argument.

---

- **Reference**: [mosfet](../../reference/device/mosfet.md)
- **Implementation**: `neurox/device/mosfet.py`
- **Tests**: `tests/test_mosfet.py`
- **Decisions**: [ADR-0002 NMOS is a pure electrical primitive](../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md), [ADR-0005 MOSFET is a polarity-parameterized electrical primitive](../../about/adr/ADR-0005-mosfet-is-a-polarity-parameterized-electrical-primitive.md)
