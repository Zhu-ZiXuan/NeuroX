# `neurox/device/rram.py`

## Current role

`RRAM` models one resistive memory device array with a continuous programmable conductance:

- programming-time non-idealities (state-dependent Gamma, stuck-at, drift)
- read-time non-idealities (telegraph, thermal)
- programmed conductance state ownership
- continuous I-V law (linear or `sinh`-nonlinear)

## Config boundary

`RRAMConfig` carries the device's intrinsic physics and per-cell parasitics:

- `g_min__uS`
- `nonlinearity_alpha`, `drift_decay_rate`, `drift_t0`
- `c_top__fF`, `c_bot__fF`
- optional `prog_gamma`, `read_telegraph`, `read_thermal`, `stuck_at`

`g_max__uS` is passed at `__init__` time as a separate kwarg; it is a design value bounded by external current limiting, not an intrinsic device parameter.

## Programming interface

`RRAM.program(target_g__uS, t_elapsed=0.0)` accepts a target-conductance tensor in the device's conductance domain. The device clamps to `[g_min__uS, g_max__uS]`, applies programming Gamma, drift, and stuck-at faults, clamps again, and stores the result in `self.g__uS`.

## State holding

`program(...)` materialises the programmed conductance state inside the `RRAM` instance. Runtime reads consume explicit `snapshot(shape=...)` samples.
