# Ye2023 calibration tools

The standard solver calibration is the only retained calibration tool:

```bash
TORCH_COMPILE_DISABLE=1 uv run python -m neurox.tools.calibrate_solver.col_bl_col_sl \
    --config validations/ye2023jssc/tools/calibrate_solver.toml --device cuda:N
```

The calibrated solver pair is `(n_outer=2, n_inner=3)`.

No cell or ADC calibration script is needed. The divider table follows from the reported RRAM resistances and the geometric-mean T1 bias; the unit-width T2 leakage and signal current come from Fig. 16. The 7 uA RS-CSA reference comes from Fig. 11 and gives a rescale factor of 14 MAC units per full-resolution code.
