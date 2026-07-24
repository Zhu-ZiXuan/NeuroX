# 1T1R cell

The 1T1R cell family places one programmable storage element in series with one access device, $\mathrm{BL} - \mathrm{storage} - V_{\mathrm{X}} - \mathrm{access} - \mathrm{SL}$, with the access device gated by the word line. At a terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ and word-line drive $V_{\mathrm{WL}}$ every 1T1R cell condenses its internal access node $V_{\mathrm{X}}$ and presents a single two-terminal branch — one branch current and two signed terminal conductances — following the [cell family](../family.md) conventions. Two branch models realize this topology: a detailed model that solves the nonlinear device stack, and a linearized model that replaces the stack with calibrated per-state conductance tables. This page states what they share; each model's branch physics is on its own page.

## Physical model

Every 1T1R cell has one internal node, the access node $V_{\mathrm{X}}$ between the storage element and the access device. The storage element conducts between the bit line $V_{\mathrm{BL}}$ and $V_{\mathrm{X}}$; the access device conducts between $V_{\mathrm{X}}$ and the source line $V_{\mathrm{SL}}$, controlled by the word-line voltage $V_{\mathrm{WL}}$. The word-line drive and the programmed weight state are inputs fixed per call. The programmed weight is an integer state index in every 1T1R model; each model derives its own state count from its configuration (the detailed model from its strictly increasing state-to-conductance ladder, the linearized model from its table rows).

The cell presents a **two-terminal branch**: $V_{\mathrm{X}}$ is condensed away inside the cell, so the cell is one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize the series stack. How the condensation is computed — a nonlinear per-cell solve or a closed-form divider — is model-specific.

## Parameters

Shared by every 1T1R model — the four per-cell node-to-ground total capacitances:

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `c_bl__fF` | per-cell node-to-ground total capacitance at the BL node | fF | $\ge 0$ | Process |
| `c_x__fF` | per-cell node-to-ground total capacitance at the internal access node $V_{\mathrm{X}}$ | fF | $\ge 0$ | Process |
| `c_sl__fF` | per-cell node-to-ground total capacitance at the SL node | fF | $\ge 0$ | Process |
| `c_wl__fF` | per-cell node-to-ground total capacitance at the WL NMOS gate node (the cell owns the gate cap; the WL wire charge belongs to the array) | fF | $\ge 0$ | Process |

Each model adds its own parameters on its page. Provenance terms are defined in [module_parameter](../../../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../../../guides/calibration/README.md); file-level schema: [config reference](../../../../../api/README.md).

## Energy model

The family shares one **node-capacitance dynamic energy** formula — the cell owns and bills the grounded-cap switching energy over all four of its nodes (BL, internal $V_{\mathrm{X}}$, SL, and the WL NMOS gate). Each node carries a per-cell node-to-ground total capacitance, and every grounded cap dissipates $E = C\,V^2$ over a full charge/discharge cycle, summed at the converged operating point:

$$E = C_{\mathrm{BL}}\,V_{\mathrm{BL}}^2 + C_{\mathrm{X}}\,V_{\mathrm{X}}^2 + C_{\mathrm{SL}}\,V_{\mathrm{SL}}^2 + C_{\mathrm{WL}}\,V_{\mathrm{WL}}^2.$$

Both models consume the four node caps identically; only $V_{\mathrm{X}}$ differs by how each model condenses its branch. The WL term is the NMOS gate cap the cell owns; the WL wire charge is billed by the owning array. Wire-segment and DC-conduction energy, together with silicon area and static leakage, lie outside the cell's energy model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl` |
| $V_{\mathrm{X}}$ | internal access node | V | `XbarCell1t1rDcop.v_x__V` |
| $V_{\mathrm{WL}}$ | word-line drive voltage at the NMOS gate (input) | V | `XbarCell1t1rSnap.v_wl__V` |
| $I$ | condensed branch current (BL $\to$ SL) | uA | `XbarCellDcop.i__uA` |
| $\partial I/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `di_dvbl__uS` |
| $\partial I/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `di_dvsl__uS` |
| $C_{\mathrm{BL}}, C_{\mathrm{X}}, C_{\mathrm{SL}}, C_{\mathrm{WL}}$ | per-cell node-to-ground total capacitances | fF | `c_bl__fF`, `c_x__fF`, `c_sl__fF`, `c_wl__fF` |

## Assumptions, scope & validity

- Every 1T1R cell has exactly one internal node ($V_{\mathrm{X}}$); the series stack is storage element then access device.
- The solve is quasi-static: it finds the DC access-node operating point and does not model transient device switching within a pulse.
- The node-capacitance energy assumes a complete $0 \to \mathrm{DC} \to 0$ charge/discharge cycle per node cap per WL pulse; every node cap is referenced to ground.
- The WL node cap the cell bills is the access-device NMOS gate; the WL routing-wire charge is not the cell's and is billed by the owning array.

TODO (domain author): the validity boundary of the lumped per-cell node-to-ground totals (coupled inter-node capacitances are not represented).

## Validation

TODO: add validation evidence for family-level branch-current and signed-conductance checks.

## References

TODO: cite the series-condensation basis.

---

- **Internals**: [cell internals](../../../../../internals/primitive/xbar/cell/_1t1r/cell.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md) (`[cim_macro.array_config.cell_config]`)
