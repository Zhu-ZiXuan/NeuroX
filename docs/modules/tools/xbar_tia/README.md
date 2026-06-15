# `neurox/tools/xbar_tia/`

Config-driven TIA design exploration for one xbar tile.

The package has one user-facing CLI — [`optimize.py`](optimize.md) — that sweeps a four-axis grid over the TIA design knobs (`opamp_gain`, `pseudo_nmos_W__um`, `pseudo_nmos_L__um`, `v_nmos_bias__V`), scores each candidate against a workload Gaussian model, and emits a ranked snippet plus per-slice transfer-curve plots.

`_common.py` provides the shared primitives:

- `build_tia(config, *, device)` — assemble an `OpAmpTIA` from a `[hardware] + sweep cell`.
- `sweep_transfer(tia, *, i_min_uA, i_max_uA, n_points, device)` — evaluate the TIA's transfer curve over an input-current range.
- `fit_to_workload(curve, workload)` — derive workload statistics (mean, span, util, saturation match) against a curve.
- `linearity_r2(curve)` — coefficient-of-determination linearity score in the workload band.

See [`optimize.md`](optimize.md) for the CLI surface and TOML schema.
