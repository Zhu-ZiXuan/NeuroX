# `neurox/analog/tia/opamp_tia.py`

## Current role

`OpAmpTIA` is the op-amp + NMOS-pseudo-resistor concrete `TIA` implementation.

It owns:

- its own TIA design/spec config;
- one internal `NMOS` pseudo-resistor;
- fabricated op-amp gain state;
- runtime clamp-driver snapshots threaded through the solver-facing protocol.

## Ownership and construction

`OpAmpTIAConfig` contains:

- the owned `NMOSConfig`
- the pseudo-NMOS sizing parameters
- TIA design parameters such as reference and bias voltages
- TIA spec parameters such as mismatch, area, leakage, latency

`OpAmpTIAPolicy(TIAPolicy)` is a structured composite policy:

- `opamp_gain_sigma: bool` — apply opamp gain mismatch at fabricate time.
- `nmos: NMOSPolicy` — sub-policy forwarded to the internal pseudo-resistor NMOS.

`OpAmpTIA.__init__(*, config, policy, name, inst_shape, dtype, T__K)` constructs its internal `NMOS` with the same `inst_shape` and forwards `policy.nmos` into it. Because `NMOS` inherits `FabricateMixin`, it is cascaded automatically by `OpAmpTIA.fabricate()`. The opamp's own `_sample_fabricate_mismatch` only refreshes `opamp_gain`.

## Solver-facing contract

`OpAmpTIA` exposes the `TIA`-family boundary-clamp surface:

- `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V)` — consumes a boundary current and an `OpAmpTIASnapshot`, returns `(v_clamp__V, dVclamp_dI__MOhm)`.
- `solve_dc(...)` — richer entry point whose result dataclass also carries the op-amp output voltage after the solve.

The 1T1R solver binds `bl_driver: TIA` and calls the `solve_clamp` form; the `solve_dc` form is the OpAmpTIA-concrete extension.
