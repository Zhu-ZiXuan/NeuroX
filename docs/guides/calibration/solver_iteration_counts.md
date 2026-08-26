# Solver iteration counts

Goal: pick the fixed iteration counts the 1T1R DC path runs at — the array solver's `n_outer` and `n_inner`, and the per-cell access-node condensation's `newton_iter_num` — so the runtime executes a fixed-trip-count graph the compiler can specialize. This is a one-shot offline job: the chip config stores the picked counts and the production path monitors nothing.

Run the two commands in this order, since the array solve consumes the branch the cell condenses:

- `neurox.tools.calibrate_cell.x1t1r` — the per-cell condensation count, plus the linearized-cell fragment extracted from the same Detail model.
- `neurox.tools.calibrate_solver.col_bl_col_sl` — the nested array solver's iteration pair.

Both are chip-parameter-free by design. Neither criterion references ADC bits, ADC range, model output, or any other downstream concern; a picked count guarantees only that the solve has converged within the numerical floor of its own iterate sequence.

## How a count is picked

Every sweep applies the same two criteria, both in the run config's `[sweep]` section beside the candidate lists.

**Step-ratio plateau — the primary criterion.** For each candidate count $n$ the tool measures the solve output $u_n$ over the whole workload and reduces adjacent candidates to one number each:

$$
\operatorname{step}_n = \max \lvert u_n - u_{n-1} \rvert,
\qquad
\operatorname{ratio}_n = \frac{\operatorname{step}_n}{\operatorname{step}_{n-1}},
$$

the maximum running over every unknown class and every position. Newton's steps fall geometrically while it converges and $\operatorname{ratio} \to 1$ once round-off dominates, so the pick is the smallest $n$ whose successor's ratio exceeds `ratio_threshold` — the point past which an extra iteration buys nothing but noise. Only the iterate sequence's self-comparison enters: there is no reference solution and no chip-tuned absolute threshold, so a chip's own floating-point behaviour is baked into the floor the detector finds.

**Relative-residual guard — the sanity check.** At the picked candidate every residual is measured against a scale the same workload supplies — the cell and wire KCL currents against $\max \lvert I_{\mathrm{cell}} \rvert$, the clamp residuals against $\max \lvert V_{\mathrm{BL\,node}} \rvert$ — and each ratio must stay under `reltol`. The array trajectory records the residual that drives each update, so its final recorded residual precedes the returned terminal state by one update and deliberately makes this guard conservative; the primary plateau criterion compares the terminal states themselves. The tolerance is methodological rather than chip-tuned: it sits safely above the accumulated fp32 round-off floor, so the guard does not false-fire under fp32 while still catching a genuinely unconverged solve. Raise it only for a chip whose wire ladders are far longer or whose signal scale is far smaller; a failing guard otherwise means the candidate range is too narrow, not that the threshold is wrong.

## Step 1 — the per-cell condensation count

The Detail cell runs its own fixed Newton on the internal-node KCL after a Pade current-divider seed. The same criteria apply, read on the per-cell internal node $V_{\mathrm{X}}$ and the per-cell KCL residual rather than on the wire and clamp unknowns. The seed lands inside the Newton basin, so this plateau arrives in very few steps.

```bash
python -m neurox.tools.calibrate_cell.x1t1r \
    --config validations/<paper>/tools/calibrate_cell.toml \
    --device cpu --output-dir log/calibration/<paper>
```

The run config carries the Detail cell fragment directly — a `cell_config` table, typically pulled from the campaign params by `_neurox_use` — plus `[grid]`, `[sweep]`, and `[runtime]`. `[grid]` is the operating grid the worst point is read at: the bit-line and source-line terminal sweep, the two word-line levels, and the nominal read point `v_bl_op__V` / `v_sl_op__V` the linearization is extracted at. The sweep walks every programmed RRAM state on that grid.

One run writes two fragments into `--output-dir`:

- `cell_detail_newton_iter_num.toml` — the margined `newton_iter_num` to merge into the Detail cell fragment.
- `cell_linear.toml` — a complete linearized-cell config: the shared physical fields copied from the Detail source, plus the per-(state, word-line level) chord conductance and bit-line-side drop fraction extracted at the nominal point, with the word-line threshold at the midpoint of the grid's two levels. Selecting the linear cell afterwards is a pure config choice — point the array's `cell_config` at this fragment with `_neurox_use`. What the extraction guarantees is specified in [1T1R linear cell](../../reference/primitive/xbar/cell/1t1r_linear.md).

A cell type with a different internal topology calibrates its own count through its own tool; this one binds to the Detail 1T1R cell.

## Step 2 — the array solver pair

The nested solver is swept on two axes in sequence: Stage A pins `n_inner` at the generous `inner_ref` and sweeps `n_outer`; Stage B pins `n_outer` at the Stage A pick and sweeps `n_inner`.

```bash
python -m neurox.tools.calibrate_solver.col_bl_col_sl \
    --config validations/<paper>/tools/calibrate_solver.toml --device cpu
```

`--plot-dir` writes one step-and-residual plot per stage. The run names the macro by file in `[macro]` — `config_files`, `config_section`, `policy_file`, `policy_section` — and locates the solver table inside that config with a dotted `[macro].solver_section` such as `array_config.solver_config`. Each candidate rebuilds a fresh macro with the swept count patched onto that table, so the tool reaches through to no concrete host topology and mutates no object. Calibrate under an all-off policy: the per-candidate rebuilds are only comparable if nothing random varies between them.

Every candidate is driven by one identical seeded workload through the macro's public `vec_mat_mul`, with the activation planes serialized over the hardware sub-phase axis exactly as the engine drives the macro. `[workload].active_rows` sets how many word lines are live per plane, the rest arriving zeroed: the macro's `max_active_num` is the production-faithful operating point and `row_num` the conservative single-plane envelope. Any in-range value is legal and none is defaulted in code, so the run config states the choice. The tool derives the macro's parallel weight-program instance shape as `(batch_w,)`.

Both criteria read the solver and cell probe channels upstream of the ADC, so the discarded output codes — and any code clipping at a conservative operating point — cannot influence the pick. The solver probe records every outer and inner iteration, while the Detail cell probe records the seed, every iterative branch evaluation, and the terminal evaluation. Aligned workload solves are reduced to a complete per-iteration worst-case trajectory, but candidate acceptance reads only where each solve currently stopped: the terminal cell residual, the wire residuals of the last inner step, and the clamp residuals of the last outer clamp event. The step delta separately compares the converged iterate of adjacent candidates. A converging solve passes through large residuals by construction, so using the whole descent as the stopping guard would reject every well-behaved solve.

Stage B can legitimately find no plateau when even the smallest inner count already sits at the floor. The tool then falls back to that smallest candidate and re-verifies the residual guard there, since Stage A's guard ran at the generous `inner_ref`; a fallback that fails the guard aborts the run rather than emitting a count.

## From the pick to the config

The recommended value is the plateau pick plus the margin the run config declares — `margin` for the cell, `outer_margin` and `inner_margin` for the two solver axes, each added after its own pick. Write the recommended values into the chip config: the array pair into `[cim_macro.array_config.solver_config]`, the cell count into `[cim_macro.array_config.cell_config]`. The array solver logs the fragment ready to paste; the cell tool logs its result and, given `--output-dir`, writes both fragments there as files.

A chip's counts belong to that chip's config and are read there, never restated here. The nested formulation the pair indexes is specified in [column BL / column SL solver](../../reference/primitive/xbar/solver/col_bl_col_sl.md), and the ownership split across one solve in [crossbar DC solve](../../system_design/xbar_solve.md).

## Working dtype

`[runtime].dtype` selects `float32` or `float64` for the whole run; there is no CLI override, so the dtype is part of the reproducible record. Calibrate at the dtype production runs at. The fp32 plateau is reached in fewer iterations than the fp64 one because its round-off floor is higher, and counts picked under fp32 remain valid in fp64, where the extra headroom only tightens the residuals. Run float64 to verify a suspicious pick.
