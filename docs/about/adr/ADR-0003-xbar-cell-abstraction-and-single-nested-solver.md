# ADR-0003: Pluggable Xbar Cell and the Single Nested Solver

## Status

Accepted

## Context

The 1T1R DC operating point was solved by two parallel formulations kept in lock-step:

- a **nested** block-Gauss-Seidel solve (inner array solve at a frozen clamp pair, outer $2\times2$ Newton on the clamp voltages), and
- a **full-Jacobian** solve that lifted every circuit unknown — including each cell's internal access node $V_{\mathrm{X}}$ — into one global block-tridiagonal Newton step ($3\times3$ diagonal blocks over the $(V_{\mathrm{BL}}, V_{\mathrm{SL}}, V_{\mathrm{X}})$ triple, the two clamp scalars Schur-eliminated against row 0).

The two were "cross-checked against each other": each was treated as evidence for the other's correctness. In practice this framing did not hold up, and the design coupling it implied was costly:

- The solver knew the cell's internal topology. Because the full-Jacobian stamped $V_{\mathrm{X}}$ and the per-cell RRAM / NMOS conductances directly into the global Jacobian, the solver was hard-wired to the 1T1R series stack. A different cell (2T1R, differential, a cell with more internal nodes) could not be introduced without rewriting the solver.
- The full-Jacobian carried a dense per-cell block-$3\times3$ Jacobian (`[..., col, row, 3, 3]`). It was faster at fp32 small batch but lost decisively on memory and generality: it OOMed at fp64 / large batch where the nested solve still ran, and its working set scaled where the nested block-$2\times2$ wire Jacobian did not.
- Mutual cross-validation is circular. Two formulations agreeing does not establish that either is correct — only that they share no *divergent* error. The nested solver is the one that runs in every production VMM and is guarded end-to-end against the physical-vs-ideal reference and independent residual checks; the full-Jacobian was never on the production path.

## Decision

### (a) Remove the full-Jacobian solver; the nested solver is the single formulation

There is now **one** 1T1R DC formulation: the nested block-Gauss-Seidel solve. It is the validated production solver — the one every VMM executes and the one held against the physical-vs-ideal reference, the independent converged-residual checks, and the chunking bit-exactness check. It is therefore the golden truth for the 1T1R operating point.

The full-Jacobian formulation is removed, for three reasons:

- **It cannot reverse-validate the production solver.** A second formulation that never runs in production and is checked only for mutual agreement adds no independent correctness signal; the nested solver's own residual and physical-reference checks are the real evidence.
- **It lost on efficiency and memory.** The dense block-$3\times3$ per-cell Jacobian OOMed at fp64 / large batch; the nested block-$2\times2$ wire Jacobian uses several times less memory and scales where the full-Jacobian could not.
- **It lost on generality.** Stamping $V_{\mathrm{X}}$ and the device conductances into a global Jacobian hard-wired the solver to the 1T1R series stack, blocking the pluggable-cell abstraction below.

### (b) Introduce the pluggable XbarCell abstraction

The cell is now a first-class, pluggable abstraction (`XbarCell`, a `FabricateMixin` + `nn.Module` + `RegistryMixin` family with config-keyed dispatch) that owns the analog two-terminal device branch and presents it to the array solver as a single condensed element:

- **Two-terminal to the solver, internal nodes condensed in the cell.** The cell exposes one BL-to-SL branch. Any internal device node — the 1T1R access node $V_{\mathrm{X}}$ between the RRAM and the access NMOS — is eliminated *inside* the cell (a Pade current-divider seed followed by a fixed number of unrolled per-cell Newton steps on the access-node KCL $F_{\mathrm{X}} = I_{\mathrm{NMOS}} - I_{\mathrm{RRAM}}$), never by the array solver. The solver drives only the wire ladders and the clamp boundaries.
- **Single consistent current.** The condensed branch reports one current (RRAM and NMOS currents agree at cell convergence; their difference is the per-cell KCL residual, exposed for diagnostics).
- **Signed terminal conductances.** The cell returns the two signed branch derivatives the wire Newton needs: $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$.
- **Control line carried in the snap.** The per-call cell snap bundles the device snaps with the cell's control-line drive (the word line for 1T1R), so the solver passes one `cell_snap` and holds no cell-internal state.
- **Per-cell-type calibration.** The per-cell Newton count `n_newton` is a calibrated numerical knob owned by the cell config, picked per cell type by the same step-ratio-plateau methodology used for the other iteration counts.
- **Device-capacitor dynamic energy owned by the cell.** The cell contributes its own internal device-capacitance switching energy (RRAM electrodes, NMOS gate / drain-body caps) per VMM; the core retains the wire-segment, control-line, and DC-conduction energy. The cell carries no PPA otherwise — its device children's area / leakage roll up through the owning core's budget.

The two decisions are one refactor: condensing the internal node inside the cell is exactly what lets the solver consume any cell through the `solve_branch` contract, which is what makes the global-Jacobian formulation both unnecessary and incompatible.

## Consequences

Positive:

- One DC formulation to maintain, calibrate, and reason about; no lock-step second solver.
- The solver is cell-agnostic: a new cell type plugs in by registering an `XbarCellConfig` subclass, with no solver change.
- Lower and better-scaling solver memory (block-$2\times2$ wire Jacobian only); fp64 / large-batch workloads that OOMed under the full-Jacobian now run.
- Clear ownership of dynamic energy: device-internal caps in the cell, wire / control / conduction in the core.

Tradeoff:

- Correctness now rests on the nested solver's own validation (independent residual checks, physical-vs-ideal agreement) rather than on cross-agreement with a second formulation. Those checks are the standing evidence and must stay green.
- Each cell type must supply its own condensation and its own calibrated `n_newton`.
