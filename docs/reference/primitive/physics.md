# Physics

The physical axioms beneath every model: the canonical SI constants (CODATA 2018), the thermal voltage $V_T = k_B T / q$, and the supply-draw laws for conduction and capacitance. The constants are fixed by definition and carry no modelling content of their own. Each subsystem applies the laws to the branches and nodes it owns.

## Conduction charge and energy

A branch carrying constant current $I$ for a conduction window $t$ transfers charge $q = It$. The energy delivered by its supply rail is

$$E = V_{\mathrm{rail}}q = V_{\mathrm{rail}}It.$$

The runtime scales close directly: $1\,\mathrm{uA}\cdot1\,\mathrm{ns}=1\,\mathrm{fC}$ and $1\,\mathrm{V}\cdot1\,\mathrm{fC}=1\,\mathrm{fJ}$.

## Capacitive supply draw

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

## Held boundaries and phases

A return-to-zero operation may establish a boundary level once, execute one or more control phases while holding it, and then release it. The establishment is one complete excursion. Each phase is another complete excursion from the held rest state to that phase's working point and back.

The rest level is the ideal one the boundary declares, never a solved quantity: an IR drop is a product of a phase, not a state the circuit parks in. The operation therefore bills the establishment once and sums the phase displacements. It neither repeats the establishment per phase nor divides it into an artificial per-phase share.

## Node capacitance

Each circuit node states the total capacitance to ground seen at that node, including its device and interconnect parasitics. The node is billed once at its own voltage excursion.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $q$ | charge transferred during a conduction window | fC | `q_conduction__fC` |
| $I$ | branch current | uA | `i__uA` |
| $t$ | conduction-window duration | ns | `duration__ns` |
| $E$ | energy delivered with charge $q$ | fJ | `e_supply_charge__fJ` |
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
- Conduction windows and capacitive excursions are billed separately; their ownership and time bases follow [energy accounting](../../conventions/notation_conventions.md#energy-accounting-basis).
