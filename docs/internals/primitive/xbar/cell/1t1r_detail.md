# 1T1R Detail cell

`XbarCell1t1rDetail` (`neurox/primitive/xbar/cell/_1t1r_detail.py`) is the nonlinear-device 1T1R model: it owns the RRAM and access-NMOS device children and condenses the access node with a per-cell Newton. The shared 1T1R substrate lives at [1T1R cell base](1t1r.md).

## Design decisions

- **Pade current-divider seed before the Newton loop.** Seeding $V_{\mathrm{X}}$ from a first-order conductance-divider split of the BL-to-SL drop lands the iterate inside the Newton basin, so a small fixed step count converges. A cold seed would need more steps or risk a bad first step on the stiff NMOS / RRAM I-V.
- **Fixed unrolled `newton_iter_num`, no convergence branch.** The Newton loop runs exactly `newton_iter_num` steps with no data-dependent stopping test. A runtime `while` on a tensor residual breaks `torch.compile` tracing; a fixed trip count keeps the Newton loop compile-safe. The count is calibrated, not guessed (next bullet).
- **`newton_iter_num` is calibrated per cell type and owned by the cell config.** It is a numerical-convergence knob, not a chip-physics parameter and not a per-source nonideality toggle, so it lives on `XbarCell1t1rDetailConfig` (calibrated by step-ratio plateau via `neurox.tools.calibrate_cell`, see [calibration guide](../../../../guides/calibration/README.md)) and never on a Policy. Each cell type calibrates its own count because the condensation it solves is its own.

## Contracts & invariants

- **Returned current is the RRAM leg.** `solve_branch` and `solve_dc` report the RRAM current `i_r` as the condensed branch current `i__uA`. `XbarCell1t1rDetailProber(RecorderBase[XbarCell1t1rDetailRecord])` is colocated with `XbarCell1t1rDetail`. When a prober is active, each `solve_dc` call computes `|i_n - i_r|` at its final $V_{\mathrm{X}}$ and submits one `XbarCell1t1rDetailRecord`; otherwise neither the absolute residual nor the record is built.
- **`program` writes only the storage device.** It maps a state-index tensor through `_state_to_g_map__uS` and programs the RRAM; the NMOS is not programmed. The state-index shape must match the cell's `inst_shape`.

## Performance & resources

The cell's per-call working set is the device snaps plus a handful of node-voltage-shaped tensors at the per-call (chunked) leading; it allocates no $V_{\mathrm{X}}$ history across Newton steps (each step overwrites the iterate functionally). `newton_iter_num` is small (single digits at fp32), so the unrolled loop adds a constant multiple of one RRAM + one NMOS `solve_dc` per cell evaluation. The condensation removes one unknown per cell from the array solve entirely: $V_{\mathrm{X}}$ never enters the array-level system, which keeps the per-instance working set small.

## Gotchas

- **`newton_iter_num` is the Detail cell's own convergence count.** It is not shared with the other Newton / iteration counts elsewhere in the pipeline; recalibrating one does not recalibrate another.

## Known limitations

- **Cell-internal convergence is verified through residuals, not a closed-form root.** The per-cell KCL residual carried by `XbarCell1t1rDetailRecord` is the standing check that the fixed `newton_iter_num` condenses $V_{\mathrm{X}}$ to the numerical floor; there is no separate analytic-root cross-check. The device-derivative signs are checked directly — the test asserts the returned conductances are non-negative / non-positive (footer's Tests).

---

- **Reference**: [Detail cell](../../../../reference/primitive/xbar/cell/1t1r_detail.md)
- **Implementation**: `neurox/primitive/xbar/cell/_1t1r_detail.py`
- **Tests**: `tests/primitive/xbar/test_cell_detail.py`, `tests/primitive/xbar/test_col_bl_col_sl.py`, `tests/primitive/device/test_mosfet.py`
