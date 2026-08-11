# Physics

The physical axioms beneath every model: the canonical SI constants (CODATA 2018), and the closed-form laws over them — the thermal voltage $V_T = k_B T / q$, and the capacitive supply-draw billing law the rest of this document specifies. The constants are fixed by definition and carry no modelling content of their own. The billing law is single: every grounded node capacitance is billed by what its supply spends moving it. This document holds that law — the atom, the rail each term is charged to, the access timing that fixes how many excursions one access contains, and how a level held across many accesses is spread over them. Each subsystem applies the law to the nodes it owns and cites it instead of restating it.

## Supply-draw atom

Moving a grounded capacitance $C$ by $\Delta V$ transfers a charge $C\,\lvert\Delta V\rvert$, and that charge leaves the supply at the supply's own potential. The energy the supply delivers is therefore

$$E = V_{\mathrm{rail}}\,C\,\lvert\Delta V\rvert,$$

in which the node's own level does not appear. The $\tfrac{1}{2} C\,\Delta V^{2}$ left on the capacitance and the equal amount burnt in the resistance of the charging path are both inside that draw, and the model does not separate them: the accounted quantity is what the supply pin delivers.

## One charging leg per excursion

A node driven from its rest level to a working level and back travels a round trip of two legs. Only one leg draws from the supply — the return leg hands its charge to ground — so an excursion is billed exactly once, at the magnitude of its displacement, whichever direction it travels first. Billing both legs, or billing the square of a level in place of a displacement, either doubles a real excursion or invents one that never happened.

## Rail attribution

Two supplies are distinguished, and a term rides the one that actually delivers its charge:

- **Conduction-path rail** — the supply behind the driver that holds the bit-line and source-line boundaries. It carries the two rail ladders and every cell node on the conduction path, the internal access node included.
- **Control rail** — the supply behind the driver of the control line. It carries the control-line ladder and the access-device gate capacitance.

The two are accounted separately even where they hold the same potential, because a term charged to the wrong rail is invisible while the two agree and wrong the moment they differ. The potentials themselves are per-scheme values declared in that scheme's configuration; this law fixes only which term rides which rail.

## Access timing

Access waveforms are return-to-zero on both axes, which is what fixes the excursion count of an access:

- **Scan axis** — zero, drive, zero, once per access.
- **Input axis** — zero, drive, hold across the full scan, zero, once per hold. A hold returns to zero before the next one is established.

An access therefore contains one complete excursion of the scanned axis, and a scan of $N$ accesses contains one complete excursion of the input axis.

## Displacement per access, establishment per hold

An access bills the displacement between the rest level a node sits at between accesses and the level it settles to during the access. The rest level is the ideal one the boundary declares, never a solved quantity: an IR drop is a product of an access, not a state the array parks in.

Where nothing is held across accesses — the input riding the scanned axis, every node returning to ground — the rest level is zero everywhere, an access is one complete excursion, and there is no establishment term at all.

Where the conduction path holds the input across a scan, the rest level is that held level. An access then bills only the displacement away from it, and establishing the hold from ground is a separate, rarer excursion of its own.

The establishment is charged by amortization rather than to one access of the scan. One hold covers exactly one full scan, so its cost divided by the accesses of that scan is what each access carries, and a complete scan bills exactly one establishment — no more and no less than the waveform contains. A scheme whose hold spans a different number of accesses states its own divisor.

## Modelling statements

- **A line belongs to the nodes it hangs on.** A stretch of interconnect carries no level of its own; its capacitance is folded into the totals of the nodes it connects, and each node is then billed once at the level that node reaches. A structure of repeated identical seats therefore states one capacitance total per node of a seat, and the ledger over those totals is the whole capacitive account — nothing sits beside it to be reconciled against.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $E$ | energy a supply delivers for one excursion | fJ | `e_cap_excursion__fJ` |
| $V_{\mathrm{rail}}$ | potential of the supply that delivers the charge | V | `v_rail__V` |
| $C$ | node-to-ground capacitance | fF | `c__fF` |
| $\Delta V$ | signed displacement between rest and working level | V | `delta_v__V` |
| $V_T$ | thermal voltage $k_B T / q$ | V | `thermal_voltage__V` |

## Assumptions and validity

- Every capacitance is lumped from its node to ground; coupling capacitance between two signal nodes is not modelled.
- Billing is quasi-static: each excursion is assumed to complete within its access, so the displacement and not the waveform shape sets the cost.
- Rest levels are ideal declared levels, so the displacement an access bills is independent of the conduction that access carries.
- Stored and dissipated energy are not separated; the whole supply draw is charged to the access that caused it.
- The law covers capacitive draw alone. Conduction energy — a branch current over its conduction window — is the separate accounting basis in [notation_conventions](../../conventions/notation_conventions.md#energy-accounting-basis).

---

- **Internals**: [physics](../../internals/primitive/physics.md)
