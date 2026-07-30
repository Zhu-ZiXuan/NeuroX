# MOSFET

`Mosfet` is a stateless-in-conduction primitive: it precomputes temperature-scaled nominals and Pelgrom sigmas at construction, resamples per-cell `_beta__uA_per_V2` / `_vth__V` state at fabricate time, and evaluates the I-V law and its three node partials in closed form. The abstract base carries the entire model; the concrete `Nmos` / `Pmos` subclasses fix only the channel polarity.

## Design decisions

- **No layout parasitics on the device.** The device holds no layout-dependent lumped-capacitance state or parameters, so a foundry primitive stays uncoupled from any one cell layout.
- **`W__um` / `L__um` are `__init__` kwargs, not config fields.** Channel geometry is a device design parameter chosen per placement, so it stays out of the frozen `MosfetConfig`, which holds process electricals and the Pelgrom matching coefficients only.
- **Polarity lives in the concrete class, not in config.** The channel polarity is a class attribute of `Nmos` (`+1`) / `Pmos` (`-1`); `MosfetConfig` / `MosfetPolicy` / `MosfetSnap` / `MosfetDcop` are polarity-free, and `beta__uA_per_V2` stays a positive magnitude with the polarity factor applied in the I-V law. The base raises on direct instantiation or a `polarity` outside `±1`.
- **Nominals and Pelgrom sigmas are precomputed once at construction.** The temperature-scaled nominals $\beta_{\mathrm{nom}}$ / $V_{\mathrm{th,nom}}$, the softplus smoothing scale, and the area-scaled sigmas $\sigma_{V_{\mathrm{th}}}$ / $\sigma_\beta$ depend only on init-time constants, so they are computed once and stored, not recomputed per solve. The `0.1` factor in $\beta_{\mathrm{nom}}$ is the cm^2→um^2 unit reconciliation that lands the result in uA/V^2.
- **Policy is separate and per-call-site constructed.** `MosfetPolicy` (two flat `bool` mismatch flags, no defaults) is loaded from its own file; with a flag off the corresponding `apply_gaussian` is the identity, so noise-off runs allocate no mismatch draw.

## Contracts & invariants

- **`_sample_fabricate_mismatch()` is re-callable and shape-stable.** Each call resamples `_beta__uA_per_V2` / `_vth__V` at `self.inst_shape`; the fabricated maps live on the device and are not mirrored elsewhere.
- **`snapshot(shape, multi_coords)` is the read path into fabricated state.** It expands the per-cell maps to the per-call `shape`, optionally advanced-indexes a chunk via `multi_coords`, and returns a `MosfetSnap`. The solve reads `beta__uA_per_V2` and `vth__V` only from the snap, never from `self`.
- **`solve_dc(vg, vd, vs, snap)` returns three node partials with fixed signs.** $\partial I/\partial V_d \ge 0$ and $\partial I/\partial V_s \le 0$ by construction for both polarities. Inputs may be scalars or tensors and broadcast against the snap.

## Performance & resources

State is two ordinary per-cell fabricated tensors at `inst_shape` plus two scalar nominal buffers. The solve is allocation-light: softplus / sigmoid on the two overdrives, no iteration. The smoothing scale and sigmas are scalars, so they do not scale with `inst_shape`.

## Gotchas

- **The softplus uses PyTorch's `beta` as the smoothing scale, not as $\beta$.** `F.softplus(x, beta=inv_smooth_scale)` passes the smoothing scale $\lambda$ (`_inv_smooth_scale__per_V`) through the framework keyword `beta`; it is unrelated to the transconductance factor `beta__uA_per_V2`. Do not conflate the two.
- **`T__K` enters at construction, not per call.** Temperature scaling and the smoothing scale are baked into the precomputed constants; changing temperature requires reconstruction, not a per-solve argument.

---

- **Reference**: [mosfet](../../../reference/primitive/device/mosfet.md)
- **Implementation**: `neurox/primitive/device/mosfet.py`
- **Tests**: `tests/primitive/device/test_mosfet.py`
