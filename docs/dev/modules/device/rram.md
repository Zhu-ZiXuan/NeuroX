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

## Construction

`RRAM.__init__(*, cfg, inst_shape, dtype, T__K, g_max__uS)`:

- `cfg` — process + noise config.
- `inst_shape` — per-instance fabrication shape (the shape of the cell array under this RRAM instance).
- `dtype`, `T__K` — runtime context.
- `g_max__uS` — design ceiling, kept outside the cfg.

## Programming interface

`RRAM.program(target_g__uS, t_elapsed)` accepts a target-conductance tensor in the device's conductance domain. The device clamps to `[g_min__uS, g_max__uS]`, applies programming Gamma, drift, and stuck-at faults, clamps again, and stores the result in `self.g__uS` via buffer reassignment.

This `program(...)` is the **device-level physical write** — it has the same name as macro/xbar-level `program(w)` but operates in conductance units against a separate semantic surface. The two coexist without conflict.

## State holding

`RRAM` inherits `FabricateMixin` so it participates in the standard cascade, but it has no static mismatch of its own — `_sample_fabricate_mismatch()` is the inherited no-op. Manufacturing variation in RRAM enters through `program(...)` (state-dependent Gamma, stuck-at faults), not through `fabricate()`. Runtime reads consume explicit `snapshot(shape=...)` samples (telegraph + thermal).
