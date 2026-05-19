# `neurox/device/nmos.py`

## Current role

`NMOS` is a pure electrical primitive.

It owns:

- continuous electrical equations
- temperature scaling
- Pelgrom-style mismatch sampling
- fabricated `beta__uA_per_V2` / `vth__V`
- runtime snapshots for electrical solve calls

It does **not** own layout-dependent lumped parasitic capacitances anymore.

## Config / init split

`NMOSConfig` contains:

- process parameters (`mu0`, `c_ox`, `vth0`, `n_factor`, temperature coefficients)
- mismatch / spec parameters (`A_vt__mV_um`, `A_beta_relative__um`)

`NMOS.__init__(*, cfg, T__K, dtype, W__um, L__um)`:

- `cfg` — process + mismatch config.
- `T__K`, `dtype` — runtime context required by every device construction.
- `W__um`, `L__um` — explicit device design parameters, passed in at construction.

Devices do not carry a profiler `name`; device PPA / dynamic energy aggregate at the circuit level. Device design parameters stay outside the device config per the project rule.

## Fabrication lifecycle

`fabricate(shape)` populates:

- `beta__uA_per_V2`
- `vth__V`

inside the module itself.

`snapshot(...)` materializes the per-call working tensors consumed by solver and circuit code. Fabricated buffers stay on the device module and are not mirrored elsewhere.

## Parasitics

Layout-dependent parasitics are out of scope for `NMOS`. Access-transistor parasitics belong in the circuit that holds the transistor: that circuit carries its own design parameters, its own lumped parasitic coefficients, and computes its own effective capacitance scalars.
