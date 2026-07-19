# 1T1R Linear cell

The Linear 1T1R cell realizes the [1T1R family topology](cell.md) as an operating-point linearization: the nonlinear device stack is replaced by calibrated per-state effective conductance tables, so the branch is a closed-form series divider. It holds no device models; every nonideality it represents is frozen into its tables at extraction time.

## Physical model

For every programmed weight state the cell stores two effective conductances — a BL-side value $g_{\mathrm{BL}}$ for the storage slot and an SL-side value $g_{\mathrm{SL}}$ for the access slot — at each of the two word-line levels (off, on). The analog word-line drive maps to a level by a threshold: the access slot counts as on when $V_{\mathrm{WL}}$ exceeds `v_wl_on_threshold__V`. The branch is the series combination of the two selected conductances around the access node $V_{\mathrm{X}}$.

## Governing equations

With the WL level selecting the table column, the branch is linear in the terminal drop:

$$G = \frac{g_{\mathrm{BL}}\, g_{\mathrm{SL}}}{g_{\mathrm{BL}} + g_{\mathrm{SL}}}, \qquad I = G\,(V_{\mathrm{BL}} - V_{\mathrm{SL}}),$$

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} = G \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} = -G \le 0,$$

and the access node is the resistive-divider value $V_{\mathrm{X}} = V_{\mathrm{BL}} - I/g_{\mathrm{BL}}$. The internal KCL of the two-conductance divider is satisfied exactly by construction, so the per-cell residual is identically zero — there is no iterative condensation and no convergence knob.

## Table semantics

The two tables are indexed by the weight-state index: row $s$ of `g_bl_table__uS` / `g_sl_table__uS` holds the `(off, on)` conductance pair of state $s$. The row count defines the cell's weight-state count (the two tables must have equal row counts; all entries are positive). The entries are **secants, not tangents**, of a detailed model at a nominal operating point $(V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}})$:

$$g_{\mathrm{BL}} = \frac{I}{V_{\mathrm{BL}}^{\mathrm{op}} - V_{\mathrm{X}}}, \qquad g_{\mathrm{SL}} = \frac{I}{V_{\mathrm{X}} - V_{\mathrm{SL}}^{\mathrm{op}}},$$

evaluated on the converged detailed branch per state and WL level, so the linear series combination reproduces the detailed branch current exactly at the extraction point. A cut-off branch draws no current and leaves both secants degenerate; such entries are floored at a tiny positive conductance. The extraction is performed by the cell-calibration tool — see the [calibration guide](../../../../../guides/calibration/solver_iteration_counts.md).

## Noise & non-idealities

The Linear cell is deterministic: it samples no device noise and carries no mismatch of its own. Whatever nonideality the extraction source included at the operating point is baked into the tables; per-call stochastic effects are outside this model.

## Parameters

In addition to the [shared family parameters](cell.md):

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `g_bl_table__uS` | per-state `(off, on)` BL-side effective conductance | uS | row count = weight-state count; entries $> 0$ | Calibrated (linearization) |
| `g_sl_table__uS` | per-state `(off, on)` SL-side effective conductance | uS | row count = `g_bl_table__uS` row count; entries $> 0$ | Calibrated (linearization) |
| `v_wl_on_threshold__V` | analog WL level above which the access slot is on | V | — | Calibrated (linearization) |

Provenance terms are defined in [module_parameter](../../../../../conventions/module_parameter.md); how to obtain values for a new chip: [calibration guide](../../../../../guides/calibration/README.md).

## Energy model

The Linear cell uses the shared family node-capacitance energy formula (the four grounded node caps of the family parameters) with $V_{\mathrm{X}}$ from the divider.

## Symbols

In addition to the [shared family symbols](cell.md):

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $g_{\mathrm{BL}}$ | BL-side (storage-slot) effective conductance | uS | `g_bl_table__uS` |
| $g_{\mathrm{SL}}$ | SL-side (access-slot) effective conductance | uS | `g_sl_table__uS` |
| $G$ | series branch conductance | uS | `di_dvbl__uS` |
| $V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}}$ | nominal extraction operating point | V | calibration-tool grid |

## Assumptions, scope & validity

- The linearization is exact only at the extraction operating point: away from it the real stack is nonlinear while this branch is linear, so the model error grows with the terminal-voltage deviation from $(V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}})$. It suits read-out schemes that clamp the array near one operating point.
- The word line is an ideal threshold switch: partial WL drives snap to off or on; sub-threshold access-device behaviour between the levels is not represented.
- The model is deterministic; it cannot represent per-call stochastic nonidealities (telegraph, thermal read noise, mismatch draws).
- The signed-conductance family invariant holds by construction ($\pm G$ with $G > 0$).

## Validation

TODO: link the evidence in [validation/xbar](../../../../../validation/README.md) — closed-form branch checks and agreement with the extraction source at the operating point.

## References

TODO.

---

- **Internals**: [Linear cell internals](../../../../../internals/primitive/xbar/cell/_1t1r/cell_linear.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md) (`[cim_macro.array_config.cell_config]`, `_neurox_class = "XbarCell1t1rLinearConfig"`)
