# Tile internals (macro.py)

Implementation notes for `IsubIadc1t1rCimMacro` — the scientific spec is [reference/macro.md](../reference/macro.md).

## Owned modules

| Attribute | Class | `inst_shape` trailing | Reporter |
| --- | --- | --- | --- |
| `core` | `XbarArray1t1r` | `()` (w_layout `(*prefix, 2*col, row)`) | yes |
| `wl_dac` | `VoltageDac` | `(row,)` | yes |
| `bl_clamp` | `VoltageDriver` | `(2, n_lane, 1)` | yes |
| `sl_driver` | `VoltageDriver` | `(2*col,)` | yes |
| `clamp_ref` | `VoltageReference` | `()` (global scalar) | yes |
| `p_mirror` | `CurrentMirror` | `(2, n_lane, 1)` | no |
| `n_mirror` | `CurrentMirror` | `(2, n_io, 1)` | no |
| `subtractor` | `CurrentSubtractor` | `(n_io, 1)` | no |
| `bl_adc` | `SarCurrentAdc` | `(n_io, 1)` | yes |
| `reference` | `CurrentReference` | `()` | yes (static only, no forward path) |

`n_lane = col_num // mux_factor` and `n_io = col_num // io_col_num` (config methods; exact divisibility enforced by `validate_sharing`). Non-reporters get no `_area_per_inst__um2` / `_leakage_per_inst__uW`; their silicon rolls up into the tile's lumped `area_per_inst__um2` / `leakage_per_inst__uW` (geometry-dependent — the TOML records the breakdown). The boundary drivers and both references are macro-owned peers of the pure array, injected into the array solve per call.

## Forward shapes (`vec_mat_mul`)

The WL planes arrive pre-expanded from the caller — the macro performs no phase expansion of its own. Any serial axis (the consuming engine's sub-phase axis, batch) rides the anonymous broadcast leading `...` through every stage: it matches no fabricated `inst_shape` axis, so it is time-serial on all hardware and per-instance right-alignment can never collide with it. The shapes below elide the inst span that rides between the leading and the listed trailing axes (empty at `inst_shape = ()`).

| Step | Shape |
| --- | --- |
| input WL planes (pre-masked by the caller) | `[..., row]` |
| `wl_dac.convert` at the weight-grid full leading | `[..., row]` (analog) |
| `core.solve_array(...).i_bl_port__uA` (one batched solve) | `[..., 2*col]` |
| unflatten + movedim (P-polarity = phys `2c`, N = `2c+1`) | `[..., 2, col]` |
| `_split_col_lanes(col_per_lane=mux_factor)` | `[..., 2, n_lane, col/n_lane]` |
| `p_mirror.replicate` | same |
| flatten + `_split_col_lanes(col_per_lane=io_col_num)` | `[..., 2, n_io, col/n_io]` |
| `n_mirror.replicate` | same |
| `subtractor.subtract(i_dl_p, i_dl_n)` | `[..., n_io, col/n_io]` x2 (magnitude, sign) |
| `bl_adc.convert` | `[..., n_io, col/n_io]` |
| `(1 - 2*sign) * magnitude`, flatten | `[..., col]` |

The `[..., col]` code tensor is the return value — leading order preserved, primitive trailing `[col]`, no trailing movedim, and no in-macro accumulation (the consuming unit reduces its own sub-phase axis in its digital domain). The lane and IO regroupings use the CimMacro `_split_col_lanes` helper (exact reshape; `ValueError` on non-divisible input).

## Array solve stage

`vec_mat_mul` converts the WL DAC once at the weight-grid full leading (`x.expand(*leading, row)` against `core.weight_grid_shape`, so per-instance drive draws and per-op DAC energy count once), snapshots `clamp_ref` once per call (its two 0-d taps broadcast onto every chunk grid, preserving chunk bit-exactness: tap 0 = BL clamp $V_{BLC}$, tap 1 = SL drive), and calls the kernel `core.solve_array` with the two boundary drivers. Chunking and the eager-island compile boundary live in the kernel array — see [array internals](../../../../../internals/primitive/xbar/array/_1t1r/array.md).

### BL-clamp lane adapter

The BL clamp is fabricated at the real device count (`inst_shape` trailing `(2, n_lane, 1)`) but the pure array snapshots its BL driver at the flat per-column shape `[..., 2*col]`. The private `_LaneGroupedClamp` adapter bridges the two while satisfying the structural `ClampDriver` role: its `snapshot` draws the wrapped `VoltageDriver` at the polarity/lane-grouped view `[..., 2, n_lane, cols_per_lane]` — the static per-DEVICE offset broadcasts onto every column the device time-serves via the trailing size-1 axis (P and N independent), while per-solve thermal noise stays per column — then regroups to the flat physical-column order `p = 2c + polarity` the solver needs; `solve_clamp` delegates unchanged. No gather indices, no wrapper circuit classes.

## Reshape-broadcast sharing

Fabricated buffers live at the real device count and broadcast onto the forward tensors purely via a trailing size-1 axis: `inst_shape = (*prefix, 2, n_lane, 1)` gives `inst_count = 2 * n_lane` and its buffer broadcasts against `[..., 2, n_lane, col/n_lane]`, so every column a device time-serves shares that device's single static draw while P and N polarities carry independent mismatch. The ADC's per-IO offsets use the same pattern at `(n_io, 1)`, and the BL clamp reaches the array through the lane adapter above. The engine's sub-phase axis (like any batch axis) sits in the anonymous broadcast leading, left of the inst prefix, so the right-aligned buffers never collide with it: static fabrication mismatch is shared across the planes a device serves (the same physical devices serve every plane — no per-plane redraw), while per-call snapshot noise samples at full shape and therefore draws independently per plane.

## Programming

`program(w)` takes `(*inst_shape, col_num, 1, row_num)` (the size-1 digit axis kept for the CimMacro contract), validates shape and the ternary digit range explicitly (a clear error instead of a downstream state-map IndexError), then scatters: magnitude to the polarity column matching the sign via `torch.where`, `torch.stack((pwg, nwg), dim=-2).flatten(-3, -2)` producing the `[..., 2*col, row]` physical state indices for `core.program` — the inverse of the forward unflatten.

## Accounting ownership

No double-count; each energy has exactly one owner. Every serial axis (the engine's sub-phase axis, batch) rides each logged tensor's leading, so each macro-owned term — data-dependent rails and the per-op bias / replica floors alike — bills once per solved WL plane automatically (bias physically flows in every plane):

| Term | Owner |
| --- | --- |
| Array conduction, wire caps, cell switching; array latency | core (`solve_array`, one event each covering all planes) |
| Clamp drop (= mirror input leg) $\sum_c (V_{DD} - V_{BL,c}) I_{BL,c} t_{pulse}$ + per-plane CMD precharge | macro (`_log_array_side_block`) |
| p-stage output rail $V_{drop} \lvert i \rvert t$ + bias floor per (instance, plane, logical column) | macro (`_log_readout_block`) |
| n-stage output rail + bias floor | macro |
| Subtractor branches (count-output-not-input): $e_{int} = t \, V_{int} (n_{hc} I_{HC} + I_{LC} + (I_{HC} - I_{LC}))$ — the third term is the ideal (pre-nonideality) difference path; $e_{out} = t \, V_{out} I_{SUB}$ with $I_{SUB}$ the post-mismatch/offset delivered magnitude | macro |
| ADC conversion energy + step latency | ADC (self-logs) |
| Readout-chain latency, `readout_latency_per_op__ns * serial_op_count` | macro |

The readout-latency parallel divisor is the OWNING back-end stage's device count — `serial_op_count = max(1, i_sub.numel() // subtractor.inst_count)` (`n_io` semantics, matching the ADC's denominator), not the macro tile `inst_count`: each back-end device serially serves its `col/n_io` columns for every plane, so the batch, plane, and column-serial factors remain in the multiplier. Kernel mirrors and subtractor log nothing (pure transports); the CurrentReference has no forward path; static clamp bias is the BL clamp's leakage share, not a dynamic term.

## Compile boundary

`vec_mat_mul` is on the compile path via the macro entry: tracing fuses the DAC/reshape/readout math and breaks only at `core.solve_array` (the kernel eager island — see [array internals](../../../../../internals/primitive/xbar/array/_1t1r/array.md)). The scheme adds no `@torch.compile` decorators.

## Validation

`IsubIadc1t1rCimMacroConfig.validate()` chains the CimMacro validate (which owns the `active_row_num` range and `row_num` divisibility guards) plus: sharing (positive knobs; strict divisibility `col_num % mux_factor == 0` and `col_num % io_col_num == 0` — the regroupings are exact reshapes, `ValueError` on a non-divisor, no clamping), supply (non-negative `v_dd__V` / `c_cmd_per_col__fF`), non-negative accounting knobs, ADC calibration (non-empty, unique, positive rescale, `adc_bits == adc_config.n_bits`, every `adc_mode` inside `[0, adc_config.mode_num)`, and full mode coverage — each ladder mode carries a record, so the macro `adc_mode_num` = `adc_config.mode_num` equals the calibrated operating-point count), and per-mode reference consistency (`adc_config.ref_levels__uA[m] == reference_config.i_refs__uA[m]` for every mode `m`, with equal mode counts, transitively extending the ADC-side per-row length guard to the reference tap rows).
