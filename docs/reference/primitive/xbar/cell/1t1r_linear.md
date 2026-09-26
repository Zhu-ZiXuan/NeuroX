# 1T1R Linear cell

The Linear 1T1R cell realizes the [1T1R family topology](1t1r.md) as an operating-point linearization: the nonlinear device stack is replaced by calibrated per-state chord-conductance and drop-fraction tables, so the branch is a division-free closed form. It holds no device models; every nonideality it represents is frozen into its tables at extraction time.

## Physical model

For every programmed weight state the cell stores two numbers at each of the two word-line levels (off, on): the total BL-to-SL branch chord conductance $g_{\mathrm{cell}}$, and the dimensionless BL-side drop fraction $r_{\mathrm{X}}$ locating the access node $V_{\mathrm{X}}$ inside the branch divider. The analog word-line drive maps to a level by a threshold: the access slot counts as on when $V_{\mathrm{WL}}$ exceeds `v_wl_on_threshold__V`.

## Governing equations

With the WL level selecting the table column, the branch is linear in the terminal drop:

$$I = g_{\mathrm{cell}}\,(V_{\mathrm{BL}} - V_{\mathrm{SL}}), \qquad \frac{\partial I}{\partial V_{\mathrm{BL}}} = g_{\mathrm{cell}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} = -g_{\mathrm{cell}} \le 0,$$

and the access node is the divider value

$$V_{\mathrm{X}} = V_{\mathrm{BL}} - r_{\mathrm{X}}\,(V_{\mathrm{BL}} - V_{\mathrm{SL}}),$$

where $r_{\mathrm{X}} = R_{\mathrm{BL}} / (R_{\mathrm{BL}} + R_{\mathrm{SL}})$ is the BL-side share of the branch resistance. Both the current and the access node are single multiplies of the terminal drop — the branch solve is division-free. The internal KCL of the branch divider is satisfied exactly by construction, so the per-cell residual is identically zero — there is no iterative condensation and no convergence knob.

## Table semantics

The four tables are flat and indexed by the weight-state index: entry $s$ of `g_cell_off_table__uS` / `g_cell_on_table__uS` holds state $s$'s chord conductance at the WL off / on level, and entry $s$ of `vx_ratio_off_table` / `vx_ratio_on_table` the matching drop fractions. The shared length defines the cell's weight-state count (all four tables must have equal length, at least one state). The entries are **chords, not tangents**, of a detailed model at a nominal operating point $(V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}})$:

$$g_{\mathrm{cell}} = \frac{I}{V_{\mathrm{BL}}^{\mathrm{op}} - V_{\mathrm{SL}}^{\mathrm{op}}}, \qquad r_{\mathrm{X}} = \frac{V_{\mathrm{BL}}^{\mathrm{op}} - V_{\mathrm{X}}}{V_{\mathrm{BL}}^{\mathrm{op}} - V_{\mathrm{SL}}^{\mathrm{op}}},$$

evaluated on the converged detailed branch per state and WL level, so the linear branch reproduces the detailed branch current and access node exactly at the extraction point. Both denominators are the fixed read span, so a cut-off branch stays well-conditioned: its chord conductance is its honest leakage value ($g_{\mathrm{cell}} = 0$ is legal — array nonsingularity is carried by the wire conductances). The extraction is performed by the cell-calibration tool — see the [calibration guide](../../../../guides/calibration/solver_tolerances.md).

## Noise & non-idealities

The Linear cell is deterministic: it samples no device noise and carries no mismatch of its own. Whatever nonideality the extraction source included at the operating point is baked into the tables; per-call stochastic effects are outside this model.

## Parameters

In addition to the [shared family parameters](1t1r.md):

| Parameter | Meaning | Unit | Constraint | [Source](../../../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| `g_cell_off_table__uS` | per-state branch chord conductance at WL off | uS | length = weight-state count ($\ge 1$); entries finite, $\ge 0$ | Calibrated (linearization) |
| `g_cell_on_table__uS` | per-state branch chord conductance at WL on | uS | length = `g_cell_off_table__uS` length; entries finite, $\ge 0$ | Calibrated (linearization) |
| `vx_ratio_off_table` | per-state BL-side drop fraction at WL off | — | length = `g_cell_off_table__uS` length; entries finite, in $[0, 1]$ | Calibrated (linearization) |
| `vx_ratio_on_table` | per-state BL-side drop fraction at WL on | — | length = `g_cell_off_table__uS` length; entries finite, in $[0, 1]$ | Calibrated (linearization) |
| `v_wl_on_threshold__V` | analog WL level above which the access slot is on | V | — | Calibrated (linearization) |

## Energy model

The Linear cell exposes the shared family node levels, with $V_{\mathrm{X}}$ taken from the drop fraction; it assigns them no capacitance or energy.

## Symbols

In addition to the [shared family symbols](1t1r.md):

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $g_{\mathrm{cell}}$ | total branch chord conductance | uS | `g_cell_off_table__uS`, `g_cell_on_table__uS`, `di_dvbl__uS` |
| $r_{\mathrm{X}}$ | BL-side drop fraction of the access-node divider | — | `vx_ratio_off_table`, `vx_ratio_on_table` |
| $V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}}$ | nominal extraction operating point | V | calibration-tool grid |

## Assumptions, scope & validity

- The linearization is exact only at the extraction operating point: away from it the real stack is nonlinear while this branch is linear, so the model error grows with the terminal-voltage deviation from $(V_{\mathrm{BL}}^{\mathrm{op}}, V_{\mathrm{SL}}^{\mathrm{op}})$. It suits read-out schemes that clamp the array near one operating point.
- The four tables and the threshold are products of one calibration run against a fixed word-line drive alphabet, and they are valid only while the deployed drive alphabet equals that calibration alphabet; the consistency is maintained by the calibration procedure, not enforced in code.
- The word line is an ideal threshold switch: partial WL drives snap to off or on; sub-threshold access-device behaviour between the levels is not represented.
- The model is deterministic; it cannot represent per-call stochastic nonidealities (telegraph, thermal read noise, mismatch draws).
- The signed-conductance family invariant holds by construction ($\pm g_{\mathrm{cell}}$ with $g_{\mathrm{cell}} \ge 0$).

## Validation

[Cell linearization](../../../../guides/calibration/solver_tolerances.md) extracts tables at declared operating points. Agreement there does not validate extrapolation across bias conditions.
