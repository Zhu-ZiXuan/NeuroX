# Glossary

The canonical term, symbol, and code name for each concept used across NeuroX subsystems — one concept, one expression.

## Hardware structure

- **cell** — one programmable circuit at an array site; owns the device topology and its terminal relations, but no array interconnect.
- **array** — the cell grid and its internal interconnect; excludes every peripheral drive, readout, conversion, and control circuit.
- **macro** — one complete digital-analog-digital VMM unit: an array together with its peripheral drive, readout, conversion, and control circuits. The macro owns the logical-to-physical weight mapping and the inverse readout combination.
- **peripheral** — a drive, readout, conversion, or control circuit belonging to a macro but lying outside its array.
- **crossbar** — a topology qualifier, written `xbar` in code identities; never a substitute noun for either array or macro.
- **WL phase axes** — the macro-selected leading axes whose Cartesian product contains the WL drive phases executed under one held BL/SL boundary state. The macro owns their meaning and passes only their positions to the array; the array uses them to group capacitive energy.

## Operations and lifecycle

- **program** — overwrite a module's programmable weight state from the representation its interface declares; may be re-called, each call replacing the previous state.
- **fabricate** — resample static manufacturing mismatch across every owned module from the unchanged nominal template; re-callable with no state accumulating across calls, orthogonal to `program`.
- **snapshot** — the verb (the `snapshot()` method) that materializes a per-call `snap` from current dynamic noise; never persisted.
- **solve_dc** — solve a circuit's DC steady state under its boundary constraints, returning a per-call `dcop`; each solving module implements its own `solve_dc`.
- **snap** — the per-call frozen container a `snapshot` produces, holding one call's materialized dynamic noise; `*Snap` suffix. Never "transient"; write "per-call".
- **dcop** — DC operating point: the per-call frozen DC steady-state solution (node voltages / branch currents) returned by a `solve_dc`; `*Dcop` suffix.
- **record** — the per-call frozen container an emitter submits to a recorder family's side channel, holding that call's collected tensors; `*Record` suffix. The item a side channel collects, never a row of the report built from it.
- **entry** — one aggregated row a reporting surface yields, holding the figures accumulated over the records collected under a reported name; never the per-call item itself, which is a `record`.

## Signal, code, and data

- **signal** — a real analog quantity a circuit carries or produces: a voltage, a current, a charge. Its name carries the physical unit suffix of that quantity.
- **code** — a real digital integer a circuit carries or produces: a converter input or output, a programmed digit, a shifted-and-added partial sum. Dimensionless, so its name carries no unit suffix.
- **data** — the abstract numeric content a signal or a code represents, independent of how it is carried; the word to use when neither realization is meant.

`digit`, `slice`, and `value` below name data, so each stays valid whichever domain realizes it.

## Value domain and slicing

- **value domain** — the integer grid a macro can physically carry: the input grid $\mathcal{X}$, the digit count $D$, and the radix $r$, all published by the macro interface for algorithm-side ranges to map onto. The algorithm-side `value_range` (the complete value an input scalar can take) is distinct from the primitive single-cell `digit_range`.
- **digit** — the level-0 value-domain unit: the integer symbol one cell carries, in radix $r$ (`digit_radix`), bounded by the per-cell `digit_range`, and realized physically as the cell conductance state (see [device/rram](../reference/primitive/device/rram.md)). `digit_count` ($D$) digits make one slice.
- **slice** — the level-1 value-domain unit (synonyms weight-slice, one-digit slice): a fixed-capacity positional piece of a value made of `digit_count` digits, with positional ratio the slice radix $R$ (`slice_radix`). The slice count ($S_w$ weight side, $S_x$ input side) is config-given, not inferred. Slices recombine into the value by the radix-weighted shift-add. See [unit family](../reference/architecture/unit/family.md) and [digital/shift_adder](../reference/primitive/digital/shift_adder.md).
- **value** — the level-2, role-neutral algorithm scalar: a weight on the weight side, an activation on the input side, decomposed into slices for the macro and recombined from them. The term itself asserts neither application role.
- **encoding (codec)** — the generic integer-to-digit-string codec over (radix, `digit_count`, policy), selected by the discriminator `"unsigned" | "true_form" | "complement" | "canonical"`; an implementation primitive with no slicer or application semantics. Its logical value range is continuous; redundant digit strings are allowed, holes are not.
- **coding scheme** — the concrete macro's rule mapping logical values and digits to physical array positions, together with the readout combination that recovers the signed result.
