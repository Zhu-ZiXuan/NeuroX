# `neurox/xbar/_1t1r/circuit_core.py`

## Current role

`CircuitCore1T1R` is the shape-independent physical core for one 1T1R tile.

It owns:

- RRAM device
- access NMOS device
- BL clamp driver (`TIA`)
- SL driver
- WL decoder + DAC
- BL / SL / WL per-line segment R / C scalars (config-time); BL / SL fabricated segment-R tensors live on the solver only
- the state-index to target-conductance lookup table
- the Newton-Raphson solver after fabrication

Physical shape is learned at `fabricate(...)` time from the state-index tensor.

## Construction

`CircuitCore1T1RConfig` carries:

- core design/spec parameters
- access-NMOS design parameters
- access-NMOS parasitic-cap densities
- `rram_g_max__uS` — maximum programmable RRAM conductance
- `state_to_g_map__uS` — state-index to target-conductance lookup table
- child configs for every owned module

The core constructs all owned modules directly from these config fields. Family-based children dispatch via `from_config(...)`. The RRAM device receives `g_max__uS` via constructor kwarg.

Cross-field validation enforces:

- `rram_g_max__uS > rram_cfg.g_min__uS`
- `state_to_g_map__uS` strictly increasing, length ≥ 2
- `state_to_g_map__uS[0] >= rram_cfg.g_min__uS`
- `state_to_g_map__uS[-1] <= rram_g_max__uS`

## Access-NMOS parasitics

The core owns the access-transistor lumped parasitics. It derives its effective parasitic scalars from:

- `access_nmos_W__um`
- `c_gs_per_um__fF`
- `c_gd_per_um__fF`
- `c_db_per_um__fF`

## Wire segments

Per-line interconnect is described by twelve flat scalar fields on `CircuitCore1T1RConfig` (no Wire module / no `c__fF_per_um` density abstraction):

- BL: `bl_first_r__MOhm`, `bl_first_c__fF`, `bl_segment_r__MOhm`, `bl_segment_c__fF`
- SL: `sl_first_r__MOhm`, `sl_first_c__fF`, `sl_segment_r__MOhm`, `sl_segment_c__fF`
- WL: `wl_first_r__MOhm`, `wl_first_c__fF`, `wl_segment_r__MOhm`, `wl_segment_c__fF`

`first_*` is the driver-to-first-cell segment; `segment_*` is repeated cell-to-cell. At `fabricate(...)` the core assembles a 1-D segment-R tensor `[first_r, segment_r, segment_r, …]` per line and hands it to the solver, which is the sole owner of the buffer. Wire tensors stay 1-D — no per-instance mismatch or expansion — so the memory cost equals one length-N tensor per line. Per-line total cap follows `first_c + (n_cells - 1) * segment_c`.

## Fabrication flow

`fabricate(w_state_idx)` takes a state-index tensor `[..., phys_col_num, row_num]`, looks each entry up through `state_to_g_map__uS` to obtain a target-conductance tensor, and hands that to `rram.program(...)`. The RRAM device applies device-level nonidealities and stores the programmed conductance. BL / SL segment-R tensors are built from the per-line config scalars and passed to the solver.

## State flow

- leaf devices fabricate and store their own static state
- the core does not re-register child state
- the core samples per-call snapshots and passes them into the solver
