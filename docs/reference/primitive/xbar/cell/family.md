# Crossbar cell family

A crossbar cell is defined by its own device topology. When an array settles a resistive path between one bit-line node and one source-line node, its selected cell supplies one condensed branch current with two signed terminal conductances. Additional paths or observables remain topology-specific. Interconnect, boundary behavior, and grid-level energy are outside the cell model.

## Physical model

Within the resistive role, a cell is the analog device branch between a bit-line node $V_{\mathrm{BL}}$ and a source-line node $V_{\mathrm{SL}}$. Internally it may hold one or more devices and internal nodes (for example a series access node); externally this path presents a **single two-terminal branch**. The cell condenses the path's internal nodes away, so the solver sees one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize that stack. The role asserts nothing about which terminal carries the input or output, nor does it exclude topology-specific paths beside this branch. Per-read exogenous control is fixed in the per-call snap together with sampled device state, so the condensation is deterministic given that snap.

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ a cell with internal node vector $\mathbf{V}_{\mathrm{int}}$ satisfies its internal KCL — the device currents into each internal node sum to zero,

$$\mathbf{F}_{\mathrm{int}}(\mathbf{V}_{\mathrm{int}};\, V_{\mathrm{BL}}, V_{\mathrm{SL}}) = \mathbf{0}.$$

Solving this for $\mathbf{V}_{\mathrm{int}}$ condenses the cell. The condensed branch then reports a **single consistent current** $I$, positive from $V_{\mathrm{BL}}$ into $V_{\mathrm{SL}}$: at convergence the currents through the two terminals agree, and that common value is the branch current. The internal-KCL residual at the returned $\mathbf{V}_{\mathrm{int}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** of the branch, the total branch derivatives with the internal nodes eliminated,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} \le 0.$$

These definite signs are an invariant of the resistive branch: raising the bit-line potential at fixed source line cannot decrease the branch current into the source line, and raising the source-line potential cannot increase it. Any physical branch built from passive / source-symmetric devices satisfies this convention. The concrete per-device origin of the signs is topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl__V` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl__V` |
| $\mathbf{V}_{\mathrm{int}}$ | internal-node voltage vector (condensed in the cell) | V | result of `ResistiveCell.solve_dc` |
| $\mathbf{F}_{\mathrm{int}}$ | internal-node KCL residual vector, where the topology has one | uA | concrete cell's own probe-channel record |
| $I$ | condensed branch current (BL $\to$ SL) | uA | `ResistiveDcop.i__uA` |
| $\partial I/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `ResistiveDcop.di_dvbl__uS` |
| $\partial I/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `ResistiveDcop.di_dvsl__uS` |

## Noise & non-idealities

The cell itself owns no static mismatch. Non-idealities enter through the cell's device children, sampled once per read into the snap so the condensation is deterministic given that snap. Each topology specifies its concrete device-noise sources in its own document.

## Energy model

The cell exposes the converged levels of its two terminals and every internal node. It assigns no node capacitance, energy, silicon area, or leakage of its own; DC-conduction energy is also outside the cell model. Which internal node levels a site presents is topology-specific.

## Assumptions, scope & validity

- The resistive role is two-terminal: it connects one bit-line node and one source-line node, with that path's internal nodes condensed inside the cell.
- The branch presents definite-sign terminal conductances, $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$, at every operating point.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a read pulse.
- Exogenous per-read control and the sampled device read state are fixed in the per-call snap, not unknowns of the condensation.

TODO (domain author): the conditions under which a cell topology can supply a definite-sign two-terminal resistive branch (e.g. monotonicity / passivity requirements on its device set), and any topology that would violate them.

## Validation

TODO: add family-level validation evidence for branch-current and signed-conductance signs, internal-KCL residuals, and finite-difference device derivatives.

## References

TODO.
