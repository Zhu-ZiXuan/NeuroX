# ye2023jssc — calibration tools

Calibration run-configs for the Ye2023 JSSC WH-2T1R CIM macro. The only run that applies to this scheme is the step1 solver iteration-count calibration; the cell lookup is filled directly from the paper and the ADC reference current comes from the paper's annotated full-scale operating point.

## Solver (`calibrate_solver.toml`)

Pins the parallel BL/SL solver's `(n_outer, n_inner)` for step1 (the BL/SL divider solve). This is a MAIN-SESSION GPU run — the solver cold-compiles per shape (~minutes); do not run it from a subagent. Check CUDA memory and utilization first, then pick a free device `cuda:N`:

```
TORCH_COMPILE_DISABLE=1 uv run python -m neurox.tools.calibrate_solver.col_bl_col_sl \
    --config validations/ye2023jssc/tools/calibrate_solver.toml --device cuda:N
```

The run sweeps `n_outer` (Stage A, `n_inner` pinned at `inner_ref`) then `n_inner` (Stage B) via step-ratio plateau detection with a relative-residual guard, and prints the recommended TOML fragment for `[cim_macro.array_config.solver_config]`. `active_rows = 32 == row_num` is the faithful input-parallel operating point (validated `1 <= active_rows <= row_num`); the parallel weight-program instance shape is derived as `(batch_w,)`.

## No cell calibration

There is NO `calibrate_cell` run for this scheme. The WH-2T1R cell is a per-state calibrated lookup (behavioral), not a Detail device model, so the Detail-cell tool does not apply. Its values come DIRECTLY from the paper:

- divider `g_cell` / `vx_ratio`: from `R_LRS`, `R_HRS`, and `R_T1 = sqrt(R_HRS * R_LRS)` (the geometric-mean bias point).
- `I_T2` table (state x input, scaled by the slice multiplier m): `(LRS, IN=1) = I_unit = 0.5 uA` measured; `(HRS, IN=1)` set so the m = 4 plane saturates the measured 30 nA per-plane bound; `(any, IN=0)` the off-cell floor derived from the 1.0 uA row leakage over the 352 radix units of a row.

## ADC

`reference_config.i_refs__uA = [[7.0]]` (the Fig.11(a) I_LSB annotation): 224 max MAC units x 0.5 uA per unit MAC = 112 uA full scale over `2**4 = 16` codes, i.e. 14 MAC units per code — which is exactly the step the ideal macro resolves over the same `[0, 223]` window at 4 bits, so `max_bits_rescale_factor = 1.0` (the rescale currency is the ideal code, not the MAC unit). That single reference is the readout's whole reference input: the RS-CSA weighs it by `2**(bits - p)` in compare phase `p`, which spans the decision ladder `c * 7.0 uA` for `c = 1 .. 15` (top decision tap 105 uA). There is no ADC calibration run — the reference follows from the paper's annotated operating point, not from a fit.
