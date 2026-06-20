# Xbar Cell Abstract Layer

## Summary / role

Every crossbar cell — the 1T1R cell, a future differential or multi-device cell — is the same thing to the array solver: a two-terminal element bridging one bit-line node and one source-line node. It presents a single condensed branch current with two signed terminal conductances; whatever internal device topology it holds is solved inside the cell and never lifted into the array solve. This document specifies that topology-agnostic family contract — the branch the solver sees, the signed-conductance convention, the per-call snap that carries exogenous control and device samples, the lean [solve_branch](#numerical-method) vs full operating-point split, and the cell-owned device-capacitor energy. The concrete topology lives in the family directories (e.g. [_1t1r/](_1t1r/README.md)); the array-level wire ladder, boundaries, and array energy belong to the consuming core.

## Physical model

A cell is the analog device branch between a bit-line node $V_{\mathrm{BL}}$ and a source-line node $V_{\mathrm{SL}}$. Internally it may hold one or more devices and internal nodes (for example a series access node); to the array solver it presents a **single two-terminal branch**. The cell condenses every internal node away, so the array solver sees one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize the entire internal stack. Each cell drives exactly one bit-line node and one source-line node — the row/column wire nodes the consuming core owns; the cell asserts nothing about which carries the input or the output. The per-read exogenous control of the cell's devices (for example a word-line / select drive) is an input to the cell, not a solver unknown: it is fixed in the per-call snap together with the sampled device read state, so the condensation is deterministic given that snap.

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ a cell with internal node vector $\mathbf{V}_{\mathrm{int}}$ satisfies its internal KCL — the device currents into each internal node sum to zero,

$$\mathbf{F}_{\mathrm{int}}(\mathbf{V}_{\mathrm{int}};\, V_{\mathrm{BL}}, V_{\mathrm{SL}}) = \mathbf{0}.$$

Solving this for $\mathbf{V}_{\mathrm{int}}$ condenses the cell. The condensed branch then reports a **single consistent current** $I$, positive from $V_{\mathrm{BL}}$ into $V_{\mathrm{SL}}$: at convergence the currents through the two terminals agree, and that common value is the branch current. The internal-KCL residual at the returned $\mathbf{V}_{\mathrm{int}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** the array wire Newton needs, the total branch derivatives with the internal nodes eliminated,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} \le 0.$$

These definite signs are the family-wide contract on which the array wire Jacobian depends: raising the bit-line potential at fixed source line cannot decrease the branch current into the source line, and raising the source-line potential cannot increase it. Any physical cell branch built from passive / source-symmetric devices satisfies the convention; a cell that violated it would corrupt the wire Jacobian rather than error. The concrete per-device origin of the signs is topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl` |
| $\mathbf{V}_{\mathrm{int}}$ | internal-node voltage vector (condensed in the cell) | V | concrete `XbarCellDCOP` subclass |
| $\mathbf{F}_{\mathrm{int}}$ | internal-node KCL residual vector | uA | `XbarCellResiduals` subclass |
| $I$ | condensed branch current (BL $\to$ SL) | uA | `XbarCellDCOP.i__uA` |
| $\partial I/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `XbarCellDCOP.di_dvbl__uS` |
| $\partial I/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `XbarCellDCOP.di_dvsl__uS` |

## Numerical method

The cell exposes two evaluations of the same condensation, splitting the solver hot loop from the diagnostic / energy path:

- **solve_branch** — the lean, compile-safe hot path. It solves the internal nodes and returns only $(I, \partial I/\partial V_{\mathrm{BL}}, \partial I/\partial V_{\mathrm{SL}})$, exactly what the array wire Newton consumes. It runs inside the consuming solver's compiled leaf, so it carries no data-dependent control flow.
- **solve_dc** — the full operating-point superset. It returns the same condensed branch quantities plus the cell's internal-node voltages, and, on request, the internal-KCL residual diagnostic. Energy and calibration call it; the hot path does not.

Both must condense identically — `solve_dc` is `solve_branch` plus the internal-node state it exposes, not a second formulation. The internal solve scheme (the root-finding method, its well-posedness, and the fixed iteration budget) is topology-specific; the family fixes only that the internal nodes are condensed within the cell and that the two evaluations agree.

## Noise & non-idealities

The cell base owns no static mismatch of its own. Non-idealities enter through the cell's device children, each gated by its own policy switch, and are sampled once per call into the snap so the condensation is deterministic given that snap. The snap is taken at the per-call broadcast shape $(\dots, \text{col}, \text{row})$ with optional chunk selection, so each device sample stays aligned with the broadcast operating point the solver drives. The concrete device-noise sources are declared by each topology and specified in [reference/device](../device/README.md).

## Parameters

The cell base carries no parameters of its own: it is an abstract contract. Each concrete cell config carries the device configs, sizing, parasitic-cap densities, programming map, and any per-cell numerical knob (such as an internal-solve iteration count) it needs. A cell config has **no PPA fields**: the cell owns its device children, whose silicon area and leakage roll up through the owning core's PPA budget, so the only physical contribution the cell makes is the per-read device-capacitor switching energy below. Provenance terms are defined in [parameter_provenance](../parameter_provenance.md).

## Energy model

The cell contributes the **device-capacitor dynamic energy** — the per-read charge/discharge energy of the capacitances internal to its own devices, summed at the converged operating point (the terminal voltages and the condensed internal-node voltages). It excludes the wire-segment, control-line, and DC-conduction energy, which the consuming core owns. The cell carries no other PPA: its device children's area and leakage roll up through the owning core's budget. The concrete capacitance inventory is topology-specific.

## Assumptions, scope & validity

Stated assumptions of the family contract:

- A cell is two-terminal to the solver: it connects exactly one bit-line node and one source-line node, with every internal node condensed inside the cell.
- The branch presents definite-sign terminal conductances, $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$, at every operating point.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a read pulse.
- Exogenous per-read control and sampled device read state are fixed in the per-call snap, not solver unknowns.

TODO (domain author): the conditions under which a candidate cell topology can be condensed to a definite-sign two-terminal branch (e.g. monotonicity / passivity requirements on its device set), and any topology that would violate the family contract.

## Validation

TODO: link [validation/xbar](../../validation/README.md) — the family-level checks any concrete cell must pass (branch-current and signed-conductance sign checks, internal-KCL residual, finite-difference device-derivative checks).

## References

TODO.

---

- **Internals**: [cell internals](../../internals/xbar/cell.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../api/README.md)
- **Decisions**: [ADR-0003 pluggable xbar cell and the single nested solver](../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
