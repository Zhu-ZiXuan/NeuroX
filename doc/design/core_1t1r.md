# Core1T1R — shape-independent physical core

`Core1T1R` (`neurox/xbar/core_1t1r.py`) is the **physical-core layer**
introduced by the 1T1R xbar refactor described in
`temp/1t1r_xbar.md`.  It owns every shape-independent physical
sub-component of one 1T1R tile and exposes a thin, encoding-agnostic
forward contract that the mapping layer (today's `Offset1T1RXbar`,
the future `Differential1T1RXbar`) consumes.

## Layered architecture

```
mapping / design layer   ─ Offset1T1RXbar (today)
                            └─ Differential1T1RXbar (future)
        ↓
core_1t1r                ─ Core1T1R
        ↓
readout                  ─ ReadOut
        ↓
analog_mux               ─ AnalogMux  (constructed; unused today)
        ↓
adc                      ─ differential ADC
        ↓
digital post-process     ─ XbarMacro digital path
```

The mapping layer owns the logical→physical weight layout (ref-column
scatter for offset coding, pos/neg column pairing for differential).
It calls `core.fabricate(w_phys)` once with the physically-shaped
weight tensor; the core derives shape from `w_phys.shape`, registers
all per-cell buffers, and binds the DC solver.  Every per-VMM call
goes through `core(x) → Core1T1ROutput(v_out_phys,
dynamic_energy__fJ)`; the mapping layer then regroups `v_out_phys`
semantically and hands the result to the readout.

## Configuration boundary

`Core1T1RConfig` (frozen, kw-only) carries only **shape-independent
physical knobs**:

| Field | Role |
|---|---|
| `wl_pulse_length__ns` | WL-high pulse window — thermal integration time. |
| `row_cell_space__um` | Pitch between adjacent cells along SL/WL. |
| `row_first_space__um` | Distance from the SL/WL driver to the first cell. |
| `col_cell_space__um` | Pitch between adjacent cells along BL. |
| `col_first_space__um` | Distance from the BL driver to the first cell. |

Out-of-config fields (kept on `Offset1T1RXbarConfig`): `col_num`,
`row_num`, `w_digit_count`, `w_digit_radix`, `w_state_offset`,
`ref_group_size`, `ref_location`.  These describe how *logical*
weights become *physical* columns and belong to the encoding, not
the array.

The bundled `default_1t1r.toml` carries the core's five fields in
a top-level `[core]` section.  The `[xbar]` section was slimmed to
the offset-coding knobs.  Tests + macro factory build the two
configs from their respective sections.

## Lifecycle

1. **`__init__(cfg, *, rram_factory, nmos_factory,
   tia_factory, *_wire, sl_driver_factory,
   wl_decoder_factory, wl_dac_factory)`** — instantiates each owned
   module via its factory (no instance shared with another core) and
   caches `v_dd_wl__V` from the WL DAC's on-level
   (`code_to_signal[1]`).  No solver yet — shape is not known, no
   per-cell buffers registered yet.  Per-segment wire resistance is
   not derived here either: it is supplied directly by
   `Core1T1RConfig`'s `bl_r_*__MOhm` / `sl_r_*__MOhm` fields (see
   `temp/wire.md`) and threaded into `Wire.fabricate(...)` in
   step 2.
2. **`fabricate(w_phys)`** — reads `phys_col_num` and `row_num`
   from `w_phys.shape`, delegates fabrication to each owned module
   (`self.rram.program(w_phys)` registers `rram.state_g__uS`
   internally; `self.nmos.fabricate(w_phys.shape)` registers
   `nmos.beta__uA_per_V2` / `nmos.vth__V`;
   `self.tia.fabricate((phys_col_num,))` registers
   `tia.opamp_gain` plus the TIA's nested-NMOS buffers),
   recomputes the four capacitance scalars (`c_wl_per_row__fF`,
   `c_bl_per_node__fF`, `c_x_per_cell__fF`, `c_gd_per_cell__fF`),
   and constructs `self.solver = NewtonRaphsonSolver1T1R(...)`.
   The solver receives the TIA via its narrow
   `clamp_driver: ClampDriver` parameter; it only calls
   `solve_clamp(...)` internally.
3. **`forward(x) → Core1T1ROutput`** — samples one runtime snapshot
   per module (`rram_snapshot`, `nmos_snapshot`, `clamp_snapshot`) via
   each module's `snapshot(...)`, drives WL via
   `wl_dac.convert(x)`, drives SL via `sl_driver.drive(...)`,
   threads the runtime snapshots through `solver.solve(...)`, re-
   invokes `self.tia.solve_dc(...)` with the *same* `clamp_snapshot`
   to obtain `v_out__V` for the readout path (the richer
   concrete-class entry point — `ClampDriver.solve_clamp` only
   surfaces the boundary tuple), and aggregates the array's per-VMM
   dynamic energy.  Output is the per-physical-column clamp output
   voltage plus the energy scalar.

The DC solver's array geometry is taken from the runtime
`rram_snapshot.g_read__uS.shape` on every `solve` call — see
`solver_1t1r.md` and `temp/state_holding.md` for the runtime-state
lifecycle.

## Dynamic-energy ownership

`Core1T1R` owns the **array-boundary** energy:

- thermal Joule over the WL pulse (`v_bl_clamp · i_bl_driver +
  v_sl_drive · i_sl_driver` × `wl_pulse_length__ns`),
- WL CV² (wire + per-cell `C_gs`),
- BL node ground caps (wire + RRAM `c_top`),
- node-X ground caps (NMOS `c_db` + RRAM `c_bot`),
- Miller-coupled `C_gd` between WL and node-X,
- SL driver per-call energy from `Driver.drive`.

The WL DAC's `convert(x)` returns an energy tensor that is
deliberately dropped — it's already counted in the WL CV² term
(`c_wl_per_row__fF · V_DD,WL²`), and double-booking would
inflate the per-array budget.

ReadOut, AnalogMux, and ADC each track their own dynamic energy
through the mapping layer's forward path; the core never sees them.

## Why shape lives outside the config

The mapping layer is the only entity that knows the final physical
shape: offset coding inserts ref columns; differential coding pairs
pos/neg cells; future multi-digit schemes further expand the column
axis.  Pinning `phys_col_num` / `row_num` on the core config would
force the mapping layer to *re-declare* the shape, opening a sync
hazard.  Pulling shape from `w_phys.shape` makes the mapping the
single source of truth.

## Capability surface

After `fabricate(w_phys)` returns, the core exposes:

| Attribute | Meaning |
|---|---|
| `fabricated_row_num: int` | `w_phys.shape[-1]` |
| `fabricated_col_num: int` | `w_phys.shape[-2]` |
| `w_states: int` | `rram.num_states` |
| `x_states: int` | `2` (binary WL gating) |
| `v_dd_wl__V: float` | WL on-level voltage |
| `solver: NewtonRaphsonSolver1T1R \| None` | DC solver instance |

The mapping layer (and external tooling like
`neurox/tools/xbar_adc_boundaries.py`) consumes the core through
these accessors plus the owned submodules (`core.rram`,
`core.nmos`, `core.tia`).  Per
`temp/state_holding.md`, fabricated buffers live **inside** the
owning modules — external code reads them through the module
attribute path (`core.rram.state_g__uS`,
`core.nmos.beta__uA_per_V2`, `core.tia.opamp_gain`, etc.), not
through proxy attributes on the core itself.
