# `neurox/xbar/_1t1r/circuit_core.py`

## Current role

`CircuitCore1T1R` is the shape-independent physical core for one 1T1R tile.

It owns:

- RRAM device
- access NMOS device
- BL clamp driver (`TIA`)
- SL driver
- WL decoder + DAC
- BL / SL / WL wire objects
- the Newton-Raphson solver after fabrication

The core is encoding-agnostic. Physical shape is learned at `fabricate(...)` time from the physical weight tensor.

## Construction

`CircuitCore1T1RConfig` carries:

- core design/spec parameters
- access-NMOS design parameters
- access-NMOS parasitic-cap densities
- child configs for every owned module

The core constructs all owned modules directly from these config fields. Only family-based children dispatch polymorphically through family `from_config(...)`.

## Access-NMOS parasitics

The core, not `NMOS`, owns the access-transistor lumped parasitics.

It derives its effective parasitic scalars from:

- `access_nmos_W__um`
- `c_gs_per_um__fF`
- `c_gd_per_um__fF`
- `c_db_per_um__fF`

This follows the current project rule that layout-dependent parasitics belong to the consuming circuit rather than the electrical primitive.

## State flow

- leaf devices fabricate and store their own static state
- the core does not re-register child state
- the core samples per-call snapshots and passes them into the solver
