# `neurox/xbar/_1t1r/circuit_core.py`

## Current role

`CircuitCore1T1R` is the shape-independent physical core for one 1T1R tile.

It owns:

- RRAM device
- access NMOS device
- BL clamp driver (`TIA`)
- SL driver
- WL DAC
- BL / SL / WL per-line segment R / C scalars (config-time); the fabricated BL / SL segment-R / segment-G tensors are registered as core buffers and threaded into the stateless solver per call
- the state-index to target-conductance lookup table
- the Newton-Raphson solver (plain stateless helper; not an `nn.Module` and not part of the core's module tree)

Physical shape is committed at `__init__` via the `w_layout_shape` argument.

## Construction

`CircuitCore1T1RConfig` carries:

- core design/spec parameters
- access-NMOS design parameters
- access-NMOS parasitic-cap densities
- `rram_g_max__uS` — maximum programmable RRAM conductance
- `state_to_g_map__uS` — state-index to target-conductance lookup table
- child configs for every owned module

`CircuitCore1T1RPolicy` is a structured composite policy with one sub-policy per child:

- `rram: RRAMPolicy`
- `nmos: NMOSPolicy` — the cell-access NMOS
- `tia: TIAPolicy` — abstract base; the concrete impl (e.g. `OpAmpTIAPolicy`) is passed by the caller. Carries only nonideality toggles (e.g. `opamp_gain_sigma`); the TIA's Newton solver knobs live in `OpAmpTIAConfig`.
- `sl_driver: DriverPolicy`
- `wl_dac: DACPolicy` — abstract base; concrete impl (e.g. `GeneralDACPolicy`) is passed

Solvers have **no Policy** — their knobs are all fixed numerical constants and live on `CircuitCore1T1RConfig.solver_config`. The core forwards each sub-policy into the matching child constructor verbatim; for the solver, it calls `Solver1T1R.from_config(config=config.solver_config, ...)` and the registry picks the concrete impl (`NestedSolver1T1R` / `FullJacobianSolver1T1R`) by config type.

`CircuitCore1T1R.__init__(*, config, policy, name, w_layout_shape, dtype, T__K)` accepts `w_layout_shape = (*prefix, phys_col_num, row_num)`. The core constructs every owned child with a derived `inst_shape` that carries `prefix` uniformly — `prefix` counts independent fabricated tile instances per the profiler contract ([`profiler_and_ppa.md`](../../../architecture/profiler_and_ppa.md)):

- `rram` / `nmos` — `inst_shape = w_layout_shape` (one per-cell mismatch sample).
- `tia` / `sl_driver` — `inst_shape = (*prefix, phys_col_num)` (per-column mismatch, one independent sample per tile).
- `wl_dac` — `inst_shape = (*prefix, row_num)`.

`CircuitCore1T1R` inherits `FabricateMixin`; the cascade refreshes the children automatically when `fabricate()` is called from above. The core itself owns no static mismatch — `_sample_fabricate_mismatch` is the default no-op. Family-based children dispatch via `from_config(...)`. The RRAM device receives `g_max__uS` via constructor kwarg.

Cross-field validation enforces:

- `rram_g_max__uS > rram_config.g_min__uS`
- `state_to_g_map__uS` strictly increasing, length ≥ 2
- `state_to_g_map__uS[0] >= rram_config.g_min__uS`
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

`first_*` is the driver-to-first-cell segment; `segment_*` is repeated cell-to-cell. At `__init__` the core assembles 1-D segment tensors per line — `[first, segment, segment, …]` for R, its reciprocal G, and the same shape for C — and registers them all as core buffers. Wire R / G are handed to the solver per call; wire C is consumed directly by `_compute_array_energy__fJ`. Wire tensors stay 1-D — no per-instance mismatch or expansion — so the memory cost equals three length-N tensors per line.

BL and SL are both column-shared (one line per column, IR drops along the row axis); their segment tensors have length `row_num`. WL is row-shared. The WL line uses a single lumped capacitance `c_wl_wire_per_row__fF = wl_first_c + (phys_col_num − 1) · wl_segment_c` in the full-settle energy model, since WL has no DC conduction path and is treated as uniform along the row.

## Programming flow

`program(w_state_idx)` takes a state-index tensor whose shape matches `self._w_layout_shape = (*prefix, phys_col_num, row_num)`, looks each entry up through `state_to_g_map__uS` to obtain a target-conductance tensor, and hands that to `rram.program(...)`. The RRAM device applies device-level non-idealities and stores the programmed conductance. BL / SL segment-R / segment-G core buffers and the solver are built at `__init__` and never rewritten.

## State flow

- leaf devices fabricate / program their own static state under the auto-cascade
- the core does not re-register child state
- the core samples per-call snapshots and passes them into the solver

## `cim_read(x) -> Tensor`

**Plain forward** — one logical CIM read (drive WL → settle to DC → return BL clamp voltage). Despite calling iterative sub-solvers inside, the method itself is not a DC-operating-point solver: it has no DCOP of its own to surface and returns just the BL clamp voltage tensor that the downstream readout will sample.

`cim_read` does:

1. **Determine the broadcast leading**: keep ``x_code`` raw; build an ``x_grid = x_code.unsqueeze(-2)`` view that adds a size-1 WL-fanout slot so ``row`` aligns with g's ``row``. Broadcast g.shape against x_grid.shape, strip trailing ``(phys_col, row)`` → ``leading``. Classify each leading position as A (x-side serial) or B (inst parallel) via ``classify_leading_positions`` (`docs/dev/architecture/chunking.md`).
2. **WL DAC convert at full leading, NO fanout slot**: ``x_dac_input = x_code.expand(*leading, row)`` is a zero-copy broadcast view (NO ``(1, row)`` trailing). ``wl_dac.convert`` sees a tensor whose trailing matches its own ``_inst_shape = (*prefix, row)`` exactly — so its numel-based serial calculation is correct, per-instance dynamic energy is counted once per instance, and ``drive_thermal__V`` samples an independent noise tensor per instance. The fanout slot is added back via ``unsqueeze(-2)`` *after* convert, so the solver sees ``(*leading, 1, row)`` as before. The DAC emits exactly one energy + one latency event per VMM.
3. Iterate chunks (A-outer / B-inner) reading `policy.solve_chunk_size_x` / `solve_chunk_size_inst`. Each chunk:
   - snapshot every fabricated device at the chunk shape;
   - run `solver.solve_dc(...)` → `Solver1T1RDCOP`;
   - run `tia.solve_dc(...)` for the boundary clamp at the converged port current;
   - call `_compute_array_energy__fJ(solver_dcop=..., v_wl_drive=...)` to fold the array's wire-cap + DC-conduction + device-cap energy into a single per-chunk scalar;
   - append the chunk's `v_out_phys` + chunk energy to per-field tensor lists;
   - **drop the chunk's heavy `solver_dcop`** at iteration end — its `v_bl_node` / `v_sl_node` / `v_x_node` / `i_cell` (`chunk_size × phys_col × row`) are freed by Python's GC before the next chunk allocates. Peak solver working set stays `O(chunk × phys_col × row)`.
4. Reassemble `v_out_phys` and chunk energies back to `(*leading, ...)`.
5. **Emit one core energy + one core latency event**: ``serial_op_count = prod(leading[p] for p in a_positions)`` — only the x-side a_positions count as serial; b_positions are parallel physical xbar instances and must not multiply the per-op latency. Same convention as every other emitting leaf (which divides ``output.numel()`` by ``inst_count`` — equivalent under the leaf's static structure; core uses the explicit a/b split because its inst classification is broadcast-determined per call).
6. Return `v_out_phys`.

No `Core1T1RDCOP` is constructed anywhere. The downstream consumer (`Offset1T1RXbar.vec_mat_mul`) needs only `v_out_phys`; everything else the chunk loop produces (port currents, node voltages, residuals) is private to the loop and used in-place to compute energy.

### Driving the inner solver standalone for debug / calibration

When tests or calibration tools need access to the inner solver's raw `Solver1T1RDCOP` (residuals, per-cell node voltages, sensitivities), they construct a standalone solver and call `solver.solve_dc(...)` directly — they do not go through `cim_read`. Two patterns in the repo:

- `tests/utils/standalone_solver_fixture.py` — solver-isolation tests build RRAM / NMOS / TIA / Driver / `Solver1T1R` independently from the chip TOML, with a synthetic mid-range RRAM g pattern. Zero dependence on `CircuitCore1T1R` / `Offset1T1RXbar`.
- `neurox/tools/solver_calibrate/_common.py::_solver_inputs_from_core` — calibration walks realistic chip workload via the standard `Offset1T1RXbar` build (RRAM g programmed from `sample_w` workload, x from `sample_x_batches`) for chip-context fidelity, then assembles `solver.solve_dc(...)` kwargs by snapshotting the already-fabricated devices on `xbar.core` and bypasses `cim_read`'s chunked-loop / energy path.

## PPA fields

`CircuitCore1T1RConfig` exposes the per-instance static-PPA pair — `area_per_inst__um2`, `leakage_per_inst__uW` — for the cell array + wire infrastructure that the core owns directly, plus its own `latency_per_op__ns: float` field for per-VMM core-side latency. Dynamic energy is computed from the solved sub-DCOPs (see *Energy accounting* below); the per-VMM event built at the end of `cim_read` carries `array_energy__fJ` for energy and `self.config.latency_per_op__ns × prod(leading[p] for p in a_positions)` for latency — see step 5 of the `cim_read` section above for why only A-side leading enters the serial count.

`wl_pulse_length__ns` stays a separate field rather than being merged into `latency_per_op__ns` because it drives wire RC charging in the energy model — it is a physical-access duration that the wire-cap full-settle integral needs as a multiplier, not an attribution-only latency.

## Energy accounting

`_compute_array_energy__fJ(*, solver_dcop, v_wl_drive)` returns per-VMM array-internal energy as a single scalar tensor of shape `[*batch]`. It takes the solver's converged `Solver1T1RDCOP` and the chunk's `v_wl_drive` (already squeezed of the WL fanout dim) as bare tensors — there is no DCOP wrapper for the core's own state because the core has none of its own.

The model assumes a full `0 → DC steady → 0` cycle for every parasitic cap over one WL pulse: total dissipation is `C · V_final²` per grounded cap and `C · (V_a − V_b)²` per coupled cap (no extra factor of 2). DC conduction is the net supply power flowing into the array boundaries — `Σ V_BL_clamp · I_BL_driver + Σ V_SL_drive · I_SL_driver`, multiplied by the WL pulse length — where the boundary currents are first-segment values `(V_drive − V_node[0]) · G_seg[0]` from the solver. By Tellegen this equals the sum of RRAM, NMOS, and BL / SL wire-R Joule losses inside the array. BL / SL wire caps use a per-segment linear-V profile, evaluated as `C · (V_L² + V_L·V_R + V_R²) / 3`. NMOS `C_gs` / `C_gd` are coupled caps to V_SL and V_X respectively; nothing is merged into a lumped WL-to-ground.
