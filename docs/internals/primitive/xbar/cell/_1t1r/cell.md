# 1T1R cell

`XbarCell1T1R` (`xbar/_1t1r/cell.py`) is the concrete `XbarCell`; the shared family and container contracts live at [xbar cell base](../README.md).

## Design decisions

- **Pade current-divider seed before the Newton loop.** Seeding $V_{\mathrm{X}}$ from a first-order conductance-divider split of the BL-to-SL drop lands the iterate inside the Newton basin, so a small fixed step count converges. A cold seed would need more steps or risk a bad first step on the stiff NMOS / RRAM I-V.
- **Fixed unrolled `n_newton`, no convergence branch.** The Newton loop runs exactly `n_newton` steps with no data-dependent stopping test. A runtime `while` on a tensor residual breaks `torch.compile` tracing; a fixed trip count keeps the Newton loop compile-safe. The count is calibrated, not guessed (next bullet).
- **`n_newton` is calibrated per cell type and owned by the cell config.** It is a numerical-convergence knob, not a chip-physics parameter and not a per-source nonideality toggle, so it lives on `XbarCell1T1RConfig` (calibrated by step-ratio plateau, see [calibration guide](../../../../../guides/calibration/README.md)) and never on a Policy. Each cell type calibrates its own count because the condensation it solves is its own.

## Contracts & invariants

- **Returned current is the RRAM leg.** `solve_branch` and `solve_dc` report the RRAM current `i_r` as the condensed branch current `i__uA`; `solve_dc(compute_residuals=True)` returns `|i_n - i_r|` as the per-cell KCL residual, the signal for whether `n_newton` is sufficient.
- **`program` writes only the storage device.** It maps a state-index tensor through `state_to_g_map__uS` and programs the RRAM; the NMOS is not programmed. The state-index shape must match the cell's `inst_shape`.

## Performance & resources

The cell's per-call working set is the device snaps plus a handful of node-voltage-shaped tensors at the per-call (chunked) leading; it allocates no $V_{\mathrm{X}}$ history across Newton steps (each step overwrites the iterate functionally). `n_newton` is small (single digits at fp32), so the unrolled loop adds a constant multiple of one RRAM + one NMOS `solve_dc` per cell evaluation. The condensation removes one unknown per cell from the array solve entirely: $V_{\mathrm{X}}$ never enters the array-level system, which keeps the per-instance working set small.

## Gotchas

- **`n_newton` is the 1T1R cell's own convergence count.** It is not shared with the other Newton / iteration counts elsewhere in the pipeline; recalibrating one does not recalibrate another.

## Known limitations

- **Cell-internal convergence is verified through residuals, not a closed-form root.** The per-cell KCL residual (`solve_dc(compute_residuals=True)`) is the standing check that the fixed `n_newton` condenses $V_{\mathrm{X}}$ to the numerical floor; there is no separate analytic-root cross-check. The device-derivative signs are checked directly — the test asserts the returned conductances are non-negative / non-positive (footer's Tests).

---

- **Reference**: [cell](../../../../../reference/primitive/xbar/cell/_1t1r/cell.md)
- **Implementation**: `neurox/primitive/xbar/cell/_1t1r.py`
- **Tests**: `tests/primitive/xbar/test_nested_solver.py`, `tests/primitive/device/test_mosfet.py`
