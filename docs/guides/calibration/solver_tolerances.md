# Cell linearization

Linear-cell extraction fits chord conductances and voltage-drop fractions from converged detailed-cell DC operating points. The extraction uses `solve_dc`, which raises if the numerical solve fails to converge. To inspect a failure, call the detailed cell's `solve_dc_trace` interface; its returned DCOP may be unconverged.

## Linear-cell extraction

Run the extraction with:

```bash
python -m neurox.tools.calibration.cell --config cell_run.toml --device cpu
```

The run file supplies the operating points, computation dtype, and detailed-cell configuration. For example, with a detailed cell stored under `[cell_config]` in a neighbouring `detail.toml`:

```toml
dtype = "float64"
v_bl_op__V = 0.3
v_sl_op__V = 0.0
v_wl_off__V = 0.0
v_wl_on__V = 0.9

[cell_config]
_neurox_use = "detail.toml:cell_config"
```

The command retains `cell_linear.toml` in its run directory. `--output` writes an additional copy for configuration reuse. The tool uses a deterministic policy and the model default temperature.

The extracted chord conductance is valid for the declared bias points and device state. Qualify its electrical approximation against the intended operating envelope separately from the runtime convergence contract.

## Validation

Use the [adaptive solve checks](../../validation/structured_while_solve.md) to distinguish convergence of the detailed equations from the validity of a fitted linear approximation.
