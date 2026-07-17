# Xbar cell abstract layer

Every crossbar cell — whatever internal device topology it holds — is a two-terminal element bridging one bit-line node and one source-line node. It presents a single condensed branch current with two signed terminal conductances; the internal device topology is solved inside the cell and never exposed at its terminals. A concrete cell realizes this model for one device topology; the array-level wire ladder, boundaries, and array energy are outside the cell.

## Physical model

A cell is the analog device branch between a bit-line node $V_{\mathrm{BL}}$ and a source-line node $V_{\mathrm{SL}}$. Internally it may hold one or more devices and internal nodes (for example a series access node); externally it presents a **single two-terminal branch**. The cell condenses every internal node away, so from outside it is one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize the entire internal stack. Each cell connects exactly one bit-line node and one source-line node, and asserts nothing about which carries the input or the output. The per-read exogenous control of the cell's devices (for example a word-line / select drive) is an input to the cell, not an unknown of the condensation: it is fixed in the per-call snap together with the sampled device read state, so the condensation is deterministic given that snap.

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ a cell with internal node vector $\mathbf{V}_{\mathrm{int}}$ satisfies its internal KCL — the device currents into each internal node sum to zero,

$$\mathbf{F}_{\mathrm{int}}(\mathbf{V}_{\mathrm{int}};\, V_{\mathrm{BL}}, V_{\mathrm{SL}}) = \mathbf{0}.$$

Solving this for $\mathbf{V}_{\mathrm{int}}$ condenses the cell. The condensed branch then reports a **single consistent current** $I$, positive from $V_{\mathrm{BL}}$ into $V_{\mathrm{SL}}$: at convergence the currents through the two terminals agree, and that common value is the branch current. The internal-KCL residual at the returned $\mathbf{V}_{\mathrm{int}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** of the branch, the total branch derivatives with the internal nodes eliminated,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} \le 0.$$

These definite signs are a family-wide invariant of the branch: raising the bit-line potential at fixed source line cannot decrease the branch current into the source line, and raising the source-line potential cannot increase it. Any physical cell branch built from passive / source-symmetric devices satisfies this convention. The concrete per-device origin of the signs is topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl` |
| $\mathbf{V}_{\mathrm{int}}$ | internal-node voltage vector (condensed in the cell) | V | concrete `XbarCellDcop` subclass |
| $\mathbf{F}_{\mathrm{int}}$ | internal-node KCL residual vector | uA | `XbarCellResiduals` subclass |
| $I$ | condensed branch current (BL $\to$ SL) | uA | `XbarCellDcop.i__uA` |
| $\partial I/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `XbarCellDcop.di_dvbl__uS` |
| $\partial I/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `XbarCellDcop.di_dvsl__uS` |

## Noise & non-idealities

The cell itself owns no static mismatch. Non-idealities enter through the cell's device children, sampled once per read into the snap so the condensation is deterministic given that snap. The concrete device-noise sources are declared by each topology and specified in [reference/device](../../device/README.md).

## Energy model

The cell contributes the **device-capacitor dynamic energy** — the per-read charge/discharge energy of the capacitances internal to its own devices, summed at the converged operating point (the terminal voltages and the condensed internal-node voltages). It excludes the wire-segment, control-line, and DC-conduction energy, which lie outside the cell. The cell carries no other PPA: its device children's silicon area and leakage roll up at the array level, not through the cell. The concrete capacitance inventory is topology-specific.

## Assumptions, scope & validity

- A cell is two-terminal: it connects exactly one bit-line node and one source-line node, with every internal node condensed inside the cell.
- The branch presents definite-sign terminal conductances, $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$, at every operating point.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a read pulse.
- Exogenous per-read control and the sampled device read state are fixed in the per-call snap, not unknowns of the condensation.

TODO (domain author): the conditions under which a candidate cell topology can be condensed to a definite-sign two-terminal branch (e.g. monotonicity / passivity requirements on its device set), and any topology that would violate them.

## Validation

TODO: link [validation/xbar](../../../../validation/README.md) — the family-level checks any concrete cell must pass (branch-current and signed-conductance sign checks, internal-KCL residual, finite-difference device-derivative checks).

## References

TODO.

---

- **Internals**: [cell internals](../../../../internals/primitive/xbar/cell/README.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../api/README.md)
